#!/usr/bin/env python3
"""Standalone query-only benchmark script for ML model selection training.

This script is derived from benchmark-vanilla.py and intentionally supports
only the get-response path. It rejects inputs that already contain response rows.

Features added beyond benchmark-vanilla.py:
1. Resume support for incomplete outputs
2. Ordered streaming JSONL writes
3. enable_thinking / enable_thinking_format request support
4. Stricter query-only input guards
5. commongen_coverage concise prompt handling
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library required. Install with: pip install openai")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # Will error if config file is used

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


ENABLE_THINKING_FORMAT_DIRECT = "direct"
ENABLE_THINKING_FORMAT_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
ENABLE_THINKING_FORMAT_CHOICES = {
    ENABLE_THINKING_FORMAT_DIRECT,
    ENABLE_THINKING_FORMAT_CHAT_TEMPLATE_KWARGS,
}


def _normalize_enable_thinking_format(value: Optional[str], context: str) -> str:
    if value is None:
        return ENABLE_THINKING_FORMAT_DIRECT

    normalized = value.strip().lower()
    if normalized not in ENABLE_THINKING_FORMAT_CHOICES:
        allowed = ", ".join(sorted(ENABLE_THINKING_FORMAT_CHOICES))
        raise ValueError(
            f"Invalid enable_thinking_format '{value}' for {context}. "
            f"Allowed values: {allowed}"
        )
    return normalized


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    endpoint: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: int = 1024
    temperature: float = 0.0
    enable_thinking: Optional[bool] = None
    enable_thinking_format: str = ENABLE_THINKING_FORMAT_DIRECT

    def get_client(self) -> OpenAI:
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        default_headers = None
        if self.headers:
            default_headers = {}
            for key, value in self.headers.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    value = os.environ.get(env_var, "")
                default_headers[key] = value

        return OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
            default_headers=default_headers,
        )


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    if yaml is None:
        print(
            "Error: PyYAML required for config files. Install with: pip install pyyaml"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    configs = []
    for model_data in data.get("models", []):
        config = ModelConfig(
            name=model_data["name"],
            endpoint=model_data.get("endpoint", "http://localhost:8000/v1"),
            api_key=model_data.get("api_key"),
            headers=model_data.get("headers"),
            max_tokens=model_data.get("max_tokens", 1024),
            temperature=model_data.get("temperature", 0.0),
            enable_thinking=model_data.get("enable_thinking"),
            enable_thinking_format=_normalize_enable_thinking_format(
                model_data.get("enable_thinking_format"),
                f"model '{model_data['name']}'",
            ),
        )
        configs.append(config)

    return configs


def create_model_configs_from_list(
    models: List[str],
    endpoint: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    enable_thinking: Optional[bool] = None,
    enable_thinking_format: str = ENABLE_THINKING_FORMAT_DIRECT,
) -> List[ModelConfig]:
    return [
        ModelConfig(
            name=model,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            enable_thinking_format=enable_thinking_format,
        )
        for model in models
    ]


@dataclass
class QueryRecord:
    """A single query record for benchmarking."""

    query: str
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    category: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single query against a model."""

    query: str
    model_name: str
    response: str
    performance: float
    response_time: float
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    category: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        result = {
            "query": self.query,
            "model_name": self.model_name,
            "response": self.response,
            "performance": self.performance,
            "response_time": self.response_time,
        }

        if self.ground_truth is not None:
            result["ground_truth"] = self.ground_truth
        if self.task_name is not None:
            result["task_name"] = self.task_name
        if self.metric is not None:
            result["metric"] = self.metric
        if self.embedding_id is not None:
            result["embedding_id"] = self.embedding_id
        if self.choices is not None:
            result["choices"] = self.choices
        if self.category is not None:
            result["category"] = self.category

        result.update(self.extra_fields)
        return result


def format_concise_query(
    query: str, metric: Optional[str] = None, choices: Optional[str] = None
) -> str:
    if metric == "em_mc":
        return (
            "Answer with ONLY the letter of the correct choice (A, B, C, or D). "
            f"Do not explain.\n\nQuestion: {query}\n\nChoices: {choices}"
        )

    if metric in ("MATH", "GSM8K"):
        return (
            f"{query}\n\nAnswer with ONLY the final number or expression. Be concise."
        )

    if metric == "code_eval":
        return (
            "Write code to solve this problem. Output ONLY the code, no "
            f"explanations:\n\n{query}"
        )

    if metric == "commongen_coverage":
        return (
            "Given the following concepts, write a coherent and natural sentence "
            f"that includes all the concepts:\n\nConcepts: {query}"
        )

    return f"Answer the following question concisely in one sentence:\n\n{query}"


def load_queries(file_path: Path, deduplicate: bool = True) -> List[QueryRecord]:
    seen_queries: Dict[str, QueryRecord] = {}
    total_records = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_records += 1

                query = data.get("query", "")
                if not query:
                    print(f"Warning: Skipping line {line_num} - no query field")
                    continue

                if deduplicate and query in seen_queries:
                    continue

                record = QueryRecord(
                    query=query,
                    ground_truth=data.get("ground_truth"),
                    task_name=data.get("task_name"),
                    metric=data.get("metric"),
                    embedding_id=data.get("embedding_id"),
                    choices=data.get("choices"),
                    category=data.get("category"),
                )

                known_fields = {
                    "query",
                    "ground_truth",
                    "task_name",
                    "metric",
                    "embedding_id",
                    "choices",
                    "model_name",
                    "response",
                    "performance",
                    "response_time",
                    "category",
                }
                for key, value in data.items():
                    if key not in known_fields:
                        record.extra_fields[key] = value

                seen_queries[query] = record

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")

    records = list(seen_queries.values())

    if deduplicate and total_records != len(records):
        print(f"Loaded {total_records} records from {file_path}")
        print(f"Extracted {len(records)} unique queries (deduplicated)")
    else:
        print(f"Loaded {len(records)} queries from {file_path}")

    categories = [r.category for r in records if r.category]
    if categories:
        from collections import Counter

        cat_counts = Counter(categories)
        print(f"Categories: {dict(cat_counts)}")

    return records


def _validate_query_only_input(file_path: Path) -> None:
    response_lines: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            if not data.get("query"):
                continue

            if "response" in data and data.get("response") is not None:
                response_lines.append(line_num)

    if response_lines:
        preview = ", ".join(str(x) for x in response_lines[:8])
        if len(response_lines) > 8:
            preview += ", ..."
        raise ValueError(
            "Query-only benchmark input must not contain response rows. "
            f"Found response fields at lines: {preview}"
        )


def evaluate_response(
    response: str,
    ground_truth: Optional[str],
    metric: Optional[str] = None,
    choices: Optional[str] = None,
) -> float:
    if ground_truth is None:
        return 0.5

    response_lower = response.lower().strip()
    truth_lower = ground_truth.lower().strip()

    if response_lower == truth_lower:
        return 1.0

    if metric == "em_mc" or choices:
        return _evaluate_multiple_choice(response, ground_truth, choices)
    elif metric == "GSM8K":
        return _evaluate_gsm8k(response, ground_truth)
    elif metric == "MATH":
        return _evaluate_math(response, ground_truth)
    elif metric == "f1_score":
        return _evaluate_f1(response, ground_truth)
    elif metric == "code_eval":
        return _evaluate_code(response, ground_truth)
    elif metric == "commongen_coverage":
        return _evaluate_commongen(response, ground_truth)
    else:
        return _evaluate_cem(response, ground_truth)


def _evaluate_multiple_choice(
    response: str, ground_truth: str, choices: Optional[str]
) -> float:
    response_text = response.strip()
    truth_upper = ground_truth.upper().strip()

    if len(truth_upper) == 1 and truth_upper in "ABCDEFGHIJ":
        parenthesis_pattern = re.findall(r"\(\s*([a-zA-Z])\s*\)", response_text)
        if parenthesis_pattern:
            found_letter = parenthesis_pattern[-1].upper()
            return 1.0 if found_letter == truth_upper else 0.0

        patterns = [
            r"(?:answer(?:\s*is)?:?\s*)([A-J])\b",
            r"(?:it['\u2019]?s|is)\s+([A-J])\b",
            r"['\u2019]s\s+([A-J])\b",
            r"\b([A-J])\s+(?:because|since|as)",
            r"(?:think|believe|choose)\s+([A-J])\b",
            r"\b([A-J])\s*[.)\]:]",
            r"^([A-J])[.)\]:\s]*$",
            r"\b([A-J])$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                found_letter = match.group(1).upper()
                if found_letter == "I" and not re.match(
                    r"^I[.)\]:\s]*$", response_text.strip(), re.IGNORECASE
                ):
                    continue
                return 1.0 if found_letter == truth_upper else 0.0

        if truth_upper != "I" and re.search(
            r"\b" + truth_upper + r"\b", response_text.upper()
        ):
            return 0.8
        if truth_upper == "I" and re.search(
            r"(?:answer|choice|option)[:\s]+I\b", response_text, re.IGNORECASE
        ):
            return 0.8

    return 0.0


def _evaluate_gsm8k(response: str, ground_truth: str) -> float:
    if "####" in ground_truth:
        ground_truth_processed = ground_truth.split("####")[-1]
    else:
        ground_truth_processed = ground_truth

    ground_truth_processed = (
        ground_truth_processed.replace(",", "")
        .replace("$", "")
        .replace(".", "")
        .strip()
    )

    numbers = re.findall(r"(\-?[0-9\.\,]+)", response)
    if not numbers:
        return 0.0

    final_answer = None
    for answer in reversed(numbers):
        if answer not in {"", "."}:
            final_answer = answer
            break

    if final_answer is None:
        return 0.0

    final_answer = final_answer.replace(",", "").replace("$", "").replace(
        ".", ""
    ).strip()
    return 1.0 if final_answer == ground_truth_processed else 0.0


def _strip_latex_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    return string.strip()


def _last_boxed_string(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def _remove_boxed(text: str) -> str:
    if "\\boxed{" in text:
        start = text.find("\\boxed{") + len("\\boxed{")
        depth = 1
        end = start
        while end < len(text) and depth > 0:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        return text[start : end - 1]
    if "\\boxed " in text:
        return text.split("\\boxed ")[-1].split()[0]
    return text


def _evaluate_math(response: str, ground_truth: str) -> float:
    gt_boxed = _last_boxed_string(ground_truth)
    if gt_boxed:
        ground_truth_processed = _remove_boxed(gt_boxed)
    else:
        ground_truth_processed = ground_truth.strip()

    try:
        response_boxed = _last_boxed_string(response)
        if response_boxed:
            response_answer = _remove_boxed(response_boxed)
            if _strip_latex_string(response_answer) == _strip_latex_string(
                ground_truth_processed
            ):
                return 1.0
    except Exception:
        pass

    gt_normalized = _strip_latex_string(ground_truth_processed)
    response_normalized = _strip_latex_string(response)
    if gt_normalized and gt_normalized in response_normalized:
        return 0.8

    try:
        gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth_processed)
        resp_nums = re.findall(r"-?\d+\.?\d*", response)
        if gt_nums and resp_nums and gt_nums[-1] in resp_nums:
            return 0.7
    except Exception:
        pass

    return 0.0


def _evaluate_f1(response: str, ground_truth: str) -> float:
    import string

    def clean_words(text: str) -> set[str]:
        text = text.lower()
        for punctuation in string.punctuation:
            text = text.replace(punctuation, " ")
        return set(text.split())

    response_words = clean_words(response)
    truth_words = clean_words(ground_truth)
    if not truth_words:
        return 0.0

    if len(truth_words) <= 2:
        truth_text = ground_truth.lower()
        for punctuation in string.punctuation:
            truth_text = truth_text.replace(punctuation, "")
        if truth_text.strip() in response.lower():
            return 1.0

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "to",
        "and",
        "or",
    }
    response_content = response_words - stopwords
    truth_content = truth_words - stopwords
    if not truth_content:
        truth_content = truth_words
    if not response_content:
        response_content = response_words

    overlap = response_content & truth_content
    if not overlap:
        return 0.0

    precision = len(overlap) / len(response_content) if response_content else 0.0
    recall = len(overlap) / len(truth_content)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _evaluate_code(response: str, ground_truth: str, timeout: int = 5) -> float:
    import signal

    code_patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"def\s+\w+\s*\([^)]*\):.*?(?=\n\n|\Z)",
    ]

    code = response
    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1) if match.lastindex else match.group(0)
            break

    try:
        if ground_truth.startswith("[") and "assert" in ground_truth:
            assertions = eval(ground_truth)
            if isinstance(assertions, list):
                passed = 0
                total = len(assertions)

                def timeout_handler(signum, frame):
                    raise TimeoutError("Code execution timed out")

                alarm_supported = hasattr(signal, "SIGALRM")

                for assertion in assertions:
                    try:
                        if alarm_supported:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(timeout)

                        local_vars = {}
                        exec(code, {}, local_vars)
                        exec(assertion, local_vars)
                        passed += 1
                    except (AssertionError, TimeoutError):
                        pass
                    except Exception:
                        pass
                    finally:
                        if alarm_supported:
                            signal.alarm(0)

                return passed / total if total > 0 else 0.0
    except Exception:
        pass

    func_match = re.search(r"def\s+(\w+)", response)
    if func_match:
        func_name = func_match.group(1)
        if func_name in ground_truth.lower():
            return 0.5

    return 0.3


def _evaluate_commongen(response: str, ground_truth: str) -> float:
    required_words = set(w.strip().lower() for w in ground_truth.split(","))
    response_lower = response.lower()
    found = sum(1 for word in required_words if word in response_lower)
    return found / len(required_words) if required_words else 0.0


def _normalize_answer(text: str) -> str:
    import string

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def _evaluate_cem(response: str, ground_truth: str) -> float:
    norm_response = _normalize_answer(response)
    norm_gt = _normalize_answer(ground_truth)
    if norm_response == norm_gt or norm_gt in norm_response:
        return 1.0
    return 0.0


def _build_resume_key(
    query: str,
    model_name: str,
    task_name: Optional[str] = None,
    metric: Optional[str] = None,
    ground_truth: Optional[str] = None,
    choices: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    payload = {
        "query": query,
        "model_name": model_name,
        "task_name": task_name,
        "metric": metric,
        "ground_truth": ground_truth,
        "choices": choices,
        "category": category,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _classify_retryable_resume_row(row: Dict[str, Any]) -> Optional[str]:
    response = row.get("response")
    if response is None:
        return "empty_response"

    response_text = str(response).strip()
    if not response_text:
        return "empty_response"
    if response_text.startswith("Error:"):
        return "error_response"
    return None


def _load_completed_resume_keys(output_path: Optional[Path]) -> Set[str]:
    if output_path is None or not output_path.exists():
        return set()

    completed: Set[str] = set()
    invalid_lines = 0
    empty_responses = 0
    error_responses = 0

    with open(output_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue

            query = row.get("query")
            model_name = row.get("model_name")
            if query is None or model_name is None:
                continue

            retry_reason = _classify_retryable_resume_row(row)
            if retry_reason == "empty_response":
                empty_responses += 1
                continue
            if retry_reason == "error_response":
                error_responses += 1
                continue

            completed.add(
                _build_resume_key(
                    query=str(query),
                    model_name=str(model_name),
                    task_name=row.get("task_name"),
                    metric=row.get("metric"),
                    ground_truth=row.get("ground_truth"),
                    choices=row.get("choices"),
                    category=row.get("category"),
                )
            )

    if invalid_lines:
        print(
            f"Warning: Ignored {invalid_lines} invalid JSON lines while loading resume state from {output_path}"
        )
    if empty_responses:
        print(
            f"Info: Found {empty_responses} empty-response rows in {output_path}; they will be re-fetched."
        )
    if error_responses:
        print(
            f"Info: Found {error_responses} Error:-response rows in {output_path}; they will be re-fetched."
        )

    return completed


def _compact_resume_output(output_path: Optional[Path]) -> None:
    if output_path is None or not output_path.exists():
        return

    retained_lines: List[str] = []
    removed_empty_responses = 0
    removed_error_responses = 0
    invalid_lines = 0

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue

            retry_reason = _classify_retryable_resume_row(row)
            if retry_reason == "empty_response":
                removed_empty_responses += 1
                continue
            if retry_reason == "error_response":
                removed_error_responses += 1
                continue

            retained_lines.append(line if line.endswith("\n") else line + "\n")

    if (
        removed_empty_responses == 0
        and removed_error_responses == 0
        and invalid_lines == 0
    ):
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for line in retained_lines:
            f.write(line)

    removed_parts = []
    if removed_empty_responses:
        removed_parts.append(f"{removed_empty_responses} empty-response rows")
    if removed_error_responses:
        removed_parts.append(f"{removed_error_responses} Error:-response rows")
    if invalid_lines:
        removed_parts.append(f"{invalid_lines} invalid JSON lines")

    print(
        f"Info: Compacted resume output {output_path}; removed "
        + ", ".join(removed_parts)
        + "."
    )


def _build_error_result(
    query: QueryRecord,
    model_name: str,
    message: str,
    response_time: float = 0.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        query=query.query,
        model_name=model_name,
        response=f"Error: {message}",
        performance=0.0,
        response_time=response_time,
        ground_truth=query.ground_truth,
        task_name=query.task_name,
        metric=query.metric,
        embedding_id=query.embedding_id,
        choices=query.choices,
        category=query.category,
        extra_fields=dict(query.extra_fields),
    )


def benchmark_query(
    model_config: ModelConfig,
    query: QueryRecord,
    concise: bool = False,
) -> BenchmarkResult:
    client = model_config.get_client()
    start_time = time.time()

    query_text = query.query
    if concise:
        query_text = format_concise_query(query.query, query.metric, query.choices)

    max_tokens = model_config.max_tokens
    if query.metric == "code_eval" and max_tokens < 256:
        max_tokens = 256

    try:
        request_kwargs: Dict[str, Any] = {
            "model": model_config.name,
            "messages": [{"role": "user", "content": query_text}],
            "max_tokens": max_tokens,
            "temperature": model_config.temperature,
        }
        if model_config.enable_thinking is not None:
            if (
                model_config.enable_thinking_format
                == ENABLE_THINKING_FORMAT_CHAT_TEMPLATE_KWARGS
            ):
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": model_config.enable_thinking
                    }
                }
            else:
                request_kwargs["extra_body"] = {
                    "enable_thinking": model_config.enable_thinking
                }

        response = client.chat.completions.create(**request_kwargs)
        response_text = response.choices[0].message.content or ""
        success = True
    except Exception as e:
        response_text = f"Error: {str(e)}"
        success = False

    response_time = time.time() - start_time

    if success:
        performance = evaluate_response(
            response_text,
            query.ground_truth,
            query.metric,
            query.choices,
        )
    else:
        performance = 0.0

    return BenchmarkResult(
        query=query.query,
        model_name=model_config.name,
        response=response_text,
        performance=performance,
        response_time=response_time,
        ground_truth=query.ground_truth,
        task_name=query.task_name,
        metric=query.metric,
        embedding_id=query.embedding_id,
        choices=query.choices,
        category=query.category,
        extra_fields=query.extra_fields,
    )


def run_benchmark(
    queries: List[QueryRecord],
    model_configs: List[ModelConfig],
    concurrency: int = 4,
    progress: bool = True,
    concise: bool = False,
    on_progress=None,
    output_path: Optional[Path] = None,
    resume: bool = False,
) -> List[BenchmarkResult]:
    all_tasks = []
    for model_config in model_configs:
        for query in queries:
            key = _build_resume_key(
                query=query.query,
                model_name=model_config.name,
                task_name=query.task_name,
                metric=query.metric,
                ground_truth=query.ground_truth,
                choices=query.choices,
                category=query.category,
            )
            all_tasks.append((query, model_config, key))

    completed_resume_keys: Set[str] = set()
    if resume:
        completed_resume_keys = _load_completed_resume_keys(output_path)
        _compact_resume_output(output_path)

    tasks = [
        (query, model_config, key)
        for query, model_config, key in all_tasks
        if not resume or key not in completed_resume_keys
    ]

    already_completed = len(all_tasks) - len(tasks)
    total_tasks = len(tasks)
    results: List[Optional[BenchmarkResult]] = [None] * total_tasks

    endpoints = set(m.endpoint for m in model_configs)
    model_names = [m.name for m in model_configs]

    print(
        f"\nBenchmarking {len(queries)} queries × {len(model_configs)} models = {total_tasks} requests"
    )
    if resume:
        print(
            f"Resume mode: {already_completed} requests already completed, {total_tasks} pending"
        )
    print(f"Models: {', '.join(model_names)}")
    if len(endpoints) == 1:
        print(f"Endpoint: {list(endpoints)[0]}")
    else:
        print(f"Endpoints: {len(endpoints)} different endpoints")
        for model_config in model_configs:
            auth_info = "with API key" if model_config.api_key else "no auth"
            print(
                f"  - {model_config.name}: {model_config.endpoint} ({auth_info})"
            )
    print(f"Concurrency: {concurrency}")
    print()

    completed = 0
    failed = 0
    written = 0
    next_to_write = 0
    finished: List[bool] = [False] * total_tasks
    task_contexts: List[Tuple[QueryRecord, str]] = [
        (query, model_config.name) for query, model_config, _ in tasks
    ]

    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "a" if resume else "w", encoding="utf-8")

    try:
        if total_tasks == 0:
            print("No pending requests. Nothing to run.")
            return []

        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures: Dict[Any, Tuple[int, QueryRecord, ModelConfig]] = {}

        for idx, (query, model_config, _task_key) in enumerate(tasks):
            future = executor.submit(benchmark_query, model_config, query, concise)
            futures[future] = (idx, query, model_config)

        pending_futures = set(futures.keys())
        pbar = tqdm(total=total_tasks, desc="Benchmarking") if progress else None

        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                idx, query, model_config = futures[future]
                if finished[idx]:
                    continue
                try:
                    result = future.result()
                except Exception as e:
                    print(f"\nError processing {model_config.name}: {e}")
                    result = _build_error_result(query, model_config.name, str(e))

                results[idx] = result
                finished[idx] = True
                completed += 1
                if result.performance == 0.0 and "Error" in result.response:
                    failed += 1
                if pbar is not None:
                    pbar.update(1)

            if output_file is not None:
                while next_to_write < total_tasks:
                    if not finished[next_to_write]:
                        break
                    pending_result = results[next_to_write]
                    if pending_result is None:
                        query_rec, model_name = task_contexts[next_to_write]
                        pending_result = _build_error_result(
                            query_rec,
                            model_name,
                            "missing result",
                        )
                        results[next_to_write] = pending_result
                    output_file.write(json.dumps(pending_result.to_jsonl_dict()) + "\n")
                    output_file.flush()
                    written += 1
                    next_to_write += 1

            if on_progress and total_tasks > 0:
                pct = int(completed * 100 / total_tasks)
                on_progress(
                    pct,
                    "Benchmarking",
                    f"{completed}/{total_tasks} queries completed ({failed} errors)",
                )

        if pbar is not None:
            pbar.close()
    finally:
        if "executor" in locals():
            executor.shutdown(wait=False, cancel_futures=True)
        if output_file is not None:
            output_file.close()

    print(f"\nCompleted: {completed}/{total_tasks} ({failed} errors)")
    if output_path is not None:
        print(f"Stream-saved {written} results to {output_path}")

    return [result for result in results if result is not None]


def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 60)
    print("  Benchmark Summary")
    print("=" * 60)

    model_stats: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if result.model_name not in model_stats:
            model_stats[result.model_name] = {
                "count": 0,
                "total_perf": 0.0,
                "total_time": 0.0,
                "successes": 0,
            }

        stats = model_stats[result.model_name]
        stats["count"] += 1
        stats["total_perf"] += result.performance
        stats["total_time"] += result.response_time
        if result.performance > 0:
            stats["successes"] += 1

    print(
        f"\n{'Model':<25} {'Queries':>8} {'Avg Perf':>10} {'Avg Time':>10} {'Success':>10}"
    )
    print("-" * 65)

    for model_name, stats in sorted(model_stats.items()):
        avg_perf = stats["total_perf"] / stats["count"] if stats["count"] > 0 else 0
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        success_rate = (
            stats["successes"] / stats["count"] * 100
            if stats["count"] > 0
            else 0
        )

        print(
            f"{model_name:<25} {stats['count']:>8} {avg_perf:>10.3f} {avg_time:>9.2f}s {success_rate:>9.1f}%"
        )

    print("=" * 60)
    print()


def run_benchmark_pipeline(
    queries_path: str,
    models_yaml_path: Optional[str] = None,
    models_list: Optional[List[str]] = None,
    endpoint: str = "http://localhost:8000/v1",
    api_key: str = "dummy",
    output_path: str = "benchmark_output.jsonl",
    concurrency: int = 4,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    enable_thinking: Optional[bool] = None,
    enable_thinking_format: Optional[str] = None,
    concise: bool = False,
    limit: int = 0,
    show_progress: bool = True,
    on_progress=None,
    resume: bool = False,
) -> List[BenchmarkResult]:
    def progress(pct, step, msg):
        if on_progress:
            on_progress(pct, step, msg)

    normalized_enable_thinking_format: Optional[str] = None
    if enable_thinking_format is not None:
        normalized_enable_thinking_format = _normalize_enable_thinking_format(
            enable_thinking_format,
            "pipeline override",
        )

    progress(5, "Starting benchmark", "Loading queries and model configs")

    qpath = Path(queries_path)
    if not qpath.exists():
        raise FileNotFoundError(f"Queries file not found: {qpath}")

    _validate_query_only_input(qpath)

    queries = load_queries(qpath)
    if not queries:
        raise ValueError("No queries loaded from file")

    if limit and limit > 0:
        original_count = len(queries)
        queries = queries[:limit]
        print(f"Limited to {len(queries)} queries (from {original_count})")

    if models_yaml_path:
        config_path = Path(models_yaml_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        model_configs = load_model_configs(config_path)
        for model_config in model_configs:
            model_config.max_tokens = max_tokens
            model_config.temperature = temperature
            if enable_thinking is not None:
                model_config.enable_thinking = enable_thinking
            if normalized_enable_thinking_format is not None:
                model_config.enable_thinking_format = normalized_enable_thinking_format
        print(f"Loaded {len(model_configs)} model configurations from {config_path}")
    elif models_list:
        model_configs = create_model_configs_from_list(
            models=models_list,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            enable_thinking_format=(
                normalized_enable_thinking_format or ENABLE_THINKING_FORMAT_DIRECT
            ),
        )
    else:
        raise ValueError("Either models_yaml_path or models_list must be provided")

    if not model_configs:
        raise ValueError("No model configurations loaded")

    total_tasks = len(queries) * len(model_configs)
    progress(
        10,
        "Running benchmark",
        f"Benchmarking {len(queries)} queries x {len(model_configs)} models = {total_tasks} requests",
    )

    if concise:
        print("Using concise prompts for faster inference")

    def benchmark_progress(pct, step, msg):
        scaled = 10 + int(pct * 80 / 100)
        progress(scaled, step, msg)

    results = run_benchmark(
        queries=queries,
        model_configs=model_configs,
        concurrency=concurrency,
        progress=show_progress,
        concise=concise,
        on_progress=benchmark_progress if on_progress else None,
        output_path=Path(output_path),
        resume=resume,
    )

    progress(92, "Finalizing output", f"Streamed {len(results)} results to {output_path}")
    progress(96, "Summary", "Generating summary")
    print_summary(results)
    return results


def _parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true/false."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark query-only inputs by fetching model responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_get_response.py --queries queries.jsonl --models llama-3.2-1b,mistral-7b
  python benchmark_get_response.py --queries queries.jsonl --model-config models.yaml --output response.jsonl
  python benchmark_get_response.py --queries queries.jsonl --model-config models.yaml --resume
        """,
    )

    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to a JSONL file containing query rows without response fields.",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names (all use same endpoint)",
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML config file with model definitions",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"),
        help="API endpoint for --models (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        help="API key for --models (uses LLM_API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output.jsonl",
        help="Output file path (must be different from --queries)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output by skipping already completed entries",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024, used with --models)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0, used with --models)",
    )
    parser.add_argument(
        "--enable-thinking",
        type=_parse_bool_arg,
        default=None,
        help="Set enable_thinking value (true/false). When provided, --enable-thinking-format is required.",
    )
    parser.add_argument(
        "--enable-thinking-format",
        type=str,
        choices=sorted(ENABLE_THINKING_FORMAT_CHOICES),
        default=None,
        help="Request payload format for --enable-thinking.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (for testing). Default: no limit",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Use concise prompts to get shorter responses (faster inference)",
    )

    args = parser.parse_args()

    if args.enable_thinking is not None and args.enable_thinking_format is None:
        parser.error(
            "--enable-thinking requires --enable-thinking-format "
            "(direct or chat_template_kwargs)"
        )

    queries_abs = Path(args.queries).expanduser().resolve()
    output_abs = Path(args.output).expanduser().resolve()
    if queries_abs == output_abs:
        print("Error: --output must be a new file and cannot be the same as --queries")
        sys.exit(1)

    models_list = None
    if args.models:
        models_list = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models_list:
            print("Error: No models specified")
            sys.exit(1)

    try:
        results = run_benchmark_pipeline(
            queries_path=args.queries,
            models_yaml_path=args.model_config,
            models_list=models_list,
            endpoint=args.endpoint,
            api_key=args.api_key,
            output_path=args.output,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            enable_thinking_format=args.enable_thinking_format,
            concise=args.concise,
            limit=args.limit or 0,
            show_progress=not args.no_progress,
            resume=args.resume,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = Path(args.output)
    print("Next steps:")
    print("  1. Add categories using VSR classifier:")
    print(
        f"     python add_category_to_training_data.py --input {output_path} --output benchmark_with_category.jsonl"
    )
    print("  2. Train models:")
    print(
        "     python train.py --data-file benchmark_with_category.jsonl --output-dir models/"
    )
    print()


if __name__ == "__main__":
    main()