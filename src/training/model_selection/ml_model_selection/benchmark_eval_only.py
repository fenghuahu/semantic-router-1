#!/usr/bin/env python3
"""Standalone evaluate-only benchmark script for existing response JSONL.

This script scores existing benchmark responses without sending benchmark
inference requests. It supports pure rule evaluation plus optional
LLM-as-a-judge modes for rescoring.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library required. Install with: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_EVAL_TIMEOUT_SECONDS = 120
ORDERED_WRITE_STALL_GRACE_SECONDS = 5
JUDGE_TIMEOUT_PLACEHOLDER_REASON = "llm judge timeout"
RULE_ONLY_METRICS = {"em_mc", "gsm8k", "math", "code_eval"}
HYBRID_ELIGIBLE_METRICS = {"cem", "f1_score", "commongen_coverage"}


@dataclass
class EvalConfig:
    """Configuration for optional LLM-as-a-judge evaluation."""

    model: str
    endpoint: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS
    concurrency: int = 2
    max_retries: int = 1
    hard_fail: bool = False
    mode: str = "rule_only"
    trigger: str = "by-metric"
    fusion: str = "weighted"
    alpha: float = 0.5
    uncertain_low: float = 0.4
    uncertain_high: float = 0.8
    rubric_version: str = "v1"

    def get_client(self) -> OpenAI:
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        return OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
            timeout=float(self.timeout_seconds),
        )


@dataclass
class QueryRecord:
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


@dataclass
class ExistingResponseRecord:
    model_name: str
    response: str
    response_time: float
    query_record: QueryRecord


def detect_evaluate_only_mode(file_path: Path) -> bool:
    query_lines: List[int] = []
    response_lines: List[int] = []
    missing_response_lines: List[int] = []

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

            query_lines.append(line_num)
            if "response" in data and data.get("response") is not None:
                response_lines.append(line_num)
            else:
                missing_response_lines.append(line_num)

    if not query_lines:
        return False

    if response_lines and missing_response_lines:
        preview = ", ".join(str(x) for x in missing_response_lines[:8])
        if len(missing_response_lines) > 8:
            preview += ", ..."
        raise ValueError(
            "Mixed input detected: some rows contain response and some do not. "
            f"Missing response at lines: {preview}"
        )

    return bool(response_lines)


def load_response_records(file_path: Path) -> List[ExistingResponseRecord]:
    records: List[ExistingResponseRecord] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

            query_text = data.get("query", "")
            if not query_text:
                print(f"Warning: Skipping line {line_num} - no query field")
                continue

            if "response" not in data or data.get("response") is None:
                raise ValueError(
                    f"Evaluate-only input requires response field at line {line_num}"
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

            query_record = QueryRecord(
                query=query_text,
                ground_truth=data.get("ground_truth"),
                task_name=data.get("task_name"),
                metric=data.get("metric"),
                embedding_id=data.get("embedding_id"),
                choices=data.get("choices"),
                category=data.get("category"),
            )

            for key, value in data.items():
                if key not in known_fields:
                    query_record.extra_fields[key] = value

            raw_rt = data.get("response_time", 0.0)
            try:
                response_time = float(raw_rt)
            except (TypeError, ValueError):
                response_time = 0.0

            records.append(
                ExistingResponseRecord(
                    model_name=str(data.get("model_name") or "unknown_model"),
                    response=str(data.get("response") or ""),
                    response_time=response_time,
                    query_record=query_record,
                )
            )

    print(f"Loaded {len(records)} response records from {file_path}")
    return records


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


def _classify_retryable_resume_row(
    row: Dict[str, Any],
    input_response_text: Optional[str] = None,
    retry_on_empty_output_with_input_response: bool = False,
) -> Optional[str]:
    if row.get("judge_reason") == JUDGE_TIMEOUT_PLACEHOLDER_REASON:
        return "timeout_placeholder"

    response = row.get("response")
    response_text = "" if response is None else str(response).strip()

    if retry_on_empty_output_with_input_response:
        normalized_input_response = (input_response_text or "").strip()
        if normalized_input_response and not response_text:
            return "missing_output_response"

    return None


def _load_completed_resume_keys(
    output_path: Optional[Path],
    input_responses_by_key: Optional[Dict[str, str]] = None,
    retry_on_empty_output_with_input_response: bool = False,
) -> Set[str]:
    if output_path is None or not output_path.exists():
        return set()

    completed: Set[str] = set()
    invalid_lines = 0
    timeout_placeholders = 0
    missing_output_responses = 0

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

            resume_key = _build_resume_key(
                query=str(query),
                model_name=str(model_name),
                task_name=row.get("task_name"),
                metric=row.get("metric"),
                ground_truth=row.get("ground_truth"),
                choices=row.get("choices"),
                category=row.get("category"),
            )

            retry_reason = _classify_retryable_resume_row(
                row,
                input_response_text=(
                    input_responses_by_key.get(resume_key)
                    if input_responses_by_key is not None
                    else None
                ),
                retry_on_empty_output_with_input_response=(
                    retry_on_empty_output_with_input_response
                ),
            )
            if retry_reason == "timeout_placeholder":
                timeout_placeholders += 1
                continue
            if retry_reason == "missing_output_response":
                missing_output_responses += 1
                continue

            completed.add(resume_key)

    if invalid_lines:
        print(
            f"Warning: Ignored {invalid_lines} invalid JSON lines while loading resume state from {output_path}"
        )
    if timeout_placeholders:
        print(
            f"Info: Found {timeout_placeholders} timeout placeholder rows in {output_path}; they will be re-evaluated."
        )
    if missing_output_responses:
        print(
            f"Info: Found {missing_output_responses} output rows with empty response but non-empty input response in {output_path}; they will be re-evaluated."
        )

    return completed


def _compact_resume_output(
    output_path: Optional[Path],
    input_responses_by_key: Optional[Dict[str, str]] = None,
    retry_on_empty_output_with_input_response: bool = False,
) -> None:
    if output_path is None or not output_path.exists():
        return

    retained_lines: List[str] = []
    removed_timeouts = 0
    removed_missing_output_responses = 0
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

            query = row.get("query")
            model_name = row.get("model_name")
            resume_key = None
            if query is not None and model_name is not None:
                resume_key = _build_resume_key(
                    query=str(query),
                    model_name=str(model_name),
                    task_name=row.get("task_name"),
                    metric=row.get("metric"),
                    ground_truth=row.get("ground_truth"),
                    choices=row.get("choices"),
                    category=row.get("category"),
                )

            retry_reason = _classify_retryable_resume_row(
                row,
                input_response_text=(
                    input_responses_by_key.get(resume_key)
                    if input_responses_by_key is not None and resume_key is not None
                    else None
                ),
                retry_on_empty_output_with_input_response=(
                    retry_on_empty_output_with_input_response
                ),
            )
            if retry_reason == "timeout_placeholder":
                removed_timeouts += 1
                continue
            if retry_reason == "missing_output_response":
                removed_missing_output_responses += 1
                continue

            retained_lines.append(json.dumps(row))

    if (
        removed_timeouts == 0
        and removed_missing_output_responses == 0
        and invalid_lines == 0
    ):
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for line in retained_lines:
            f.write(line + "\n")

    removed_parts = [f"removed {removed_timeouts} timeout placeholders"]
    if removed_missing_output_responses:
        removed_parts.append(
            f"{removed_missing_output_responses} empty-output rows with non-empty input response"
        )
    if invalid_lines:
        removed_parts.append(f"{invalid_lines} invalid JSON lines")

    print(
        f"Info: Compacted resume output {output_path}; "
        + ", ".join(removed_parts)
        + "."
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
    if metric == "GSM8K":
        return _evaluate_gsm8k(response, ground_truth)
    if metric == "MATH":
        return _evaluate_math(response, ground_truth)
    if metric == "f1_score":
        return _evaluate_f1(response, ground_truth)
    if metric == "code_eval":
        return _evaluate_code(response, ground_truth)
    if metric == "commongen_coverage":
        return _evaluate_commongen(response, ground_truth)
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

    final_answer = (
        final_answer.replace(",", "").replace("$", "").replace(".", "").strip()
    )
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

                        local_vars: Dict[str, Any] = {}
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
    if func_match and func_match.group(1) in ground_truth.lower():
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


def _normalize_metric(metric: Optional[str]) -> str:
    return (metric or "cem").strip().lower()


def _should_use_judge(
    eval_config: EvalConfig,
    metric: Optional[str],
    rule_score: float,
) -> bool:
    if eval_config.mode == "judge_only":
        return True
    if eval_config.mode == "judge_review":
        return rule_score != 1.0
    if eval_config.mode == "rule_only":
        return False
    if eval_config.trigger == "always":
        return True
    if eval_config.trigger == "uncertain":
        return eval_config.uncertain_low <= rule_score <= eval_config.uncertain_high

    metric_name = _normalize_metric(metric)
    if metric_name in RULE_ONLY_METRICS:
        return False
    if metric_name in HYBRID_ELIGIBLE_METRICS:
        return True
    return True


def _build_judge_prompt(
    query: str,
    ground_truth: str,
    model_response: str,
    metric: Optional[str],
    rubric_version: str,
) -> str:
    metric_name = metric or "cem"
    return (
        "You are an objective evaluator. Score the model response against ground truth.\n"
        "Output STRICT JSON only with this schema:\n"
        '{"score": <0..1>, "label": "correct|partial|incorrect|unknown", '
        '"reason": "short reason", "confidence": <0..1>}\n'
        "No markdown. No extra keys.\n"
        f"rubric_version={rubric_version}\n"
        f"metric={metric_name}\n\n"
        f"query:\n{query}\n\n"
        f"ground_truth:\n{ground_truth}\n\n"
        f"model_response:\n{model_response}\n"
    )


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _parse_judge_result(raw_text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "judge_schema_violation": False,
        "judge_raw": raw_text,
    }

    payload = _extract_first_json_object(raw_text)
    if payload is None:
        metadata["judge_error"] = "judge_output_parse_failed"
        return None, metadata

    score = payload.get("score")
    label = payload.get("label", "unknown")
    reason = payload.get("reason", "")
    confidence = payload.get("confidence", 0.0)

    try:
        score = float(score)
    except Exception:
        metadata["judge_error"] = "judge_output_invalid_score"
        return None, metadata

    if score < 0.0 or score > 1.0:
        metadata["judge_schema_violation"] = True
        score = min(1.0, max(0.0, score))

    label = str(label).strip().lower()
    if label not in {"correct", "partial", "incorrect", "unknown"}:
        metadata["judge_schema_violation"] = True
        label = "unknown"

    try:
        confidence = float(confidence)
    except Exception:
        metadata["judge_schema_violation"] = True
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    metadata["judge_label"] = label
    metadata["judge_reason"] = str(reason)[:500]
    metadata["judge_confidence"] = confidence
    return score, metadata


def _fuse_scores(
    eval_config: EvalConfig,
    rule_score: float,
    judge_score: Optional[float],
) -> Tuple[float, str]:
    if judge_score is None:
        return rule_score, "rule_fallback"
    if eval_config.mode == "judge_only":
        return judge_score, "judge_only"
    if eval_config.mode == "judge_review":
        return judge_score, "judge_review"
    if eval_config.mode == "rule_only":
        return rule_score, "rule_only"
    if eval_config.fusion == "rule":
        return rule_score, "rule"
    if eval_config.fusion == "judge":
        return judge_score, "judge"
    if eval_config.fusion == "max":
        return max(rule_score, judge_score), "max"
    if eval_config.fusion == "min":
        return min(rule_score, judge_score), "min"

    final_score = eval_config.alpha * rule_score + (1.0 - eval_config.alpha) * judge_score
    return final_score, "weighted"


def _evaluate_with_judge(
    eval_client: OpenAI,
    eval_config: EvalConfig,
    query: QueryRecord,
    response_text: str,
    judge_semaphore: Optional[threading.Semaphore],
) -> Tuple[Optional[float], Dict[str, Any]]:
    prompt = _build_judge_prompt(
        query=query.query,
        ground_truth=query.ground_truth or "",
        model_response=response_text,
        metric=query.metric,
        rubric_version=eval_config.rubric_version,
    )

    metadata: Dict[str, Any] = {}
    last_error = None
    attempts = max(1, eval_config.max_retries + 1)
    judge_start_time = time.time()

    for attempt in range(attempts):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": eval_config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": eval_config.temperature,
                "extra_body": {
                    "enable_thinking": True,
                    "thinking_budget": 4096,
                },
            }
            judge_context = (
                judge_semaphore if judge_semaphore is not None else nullcontext()
            )
            with judge_context:
                judge_resp = eval_client.chat.completions.create(**request_kwargs)

            judge_text = judge_resp.choices[0].message.content or ""
            judge_score, parse_meta = _parse_judge_result(judge_text)
            metadata.update(parse_meta)
            metadata["judge_time"] = time.time() - judge_start_time
            return judge_score, metadata
        except Exception as e:
            last_error = str(e)
            if attempt + 1 >= attempts:
                break

    metadata["judge_error"] = f"judge_request_failed: {last_error}"
    metadata["judge_time"] = time.time() - judge_start_time
    return None, metadata


def score_existing_response(
    query: QueryRecord,
    response_text: str,
    eval_config: Optional[EvalConfig] = None,
    eval_client: Optional[OpenAI] = None,
    judge_semaphore: Optional[threading.Semaphore] = None,
) -> Tuple[float, Dict[str, Any]]:
    extra_fields = dict(query.extra_fields)

    rule_score = evaluate_response(
        response_text,
        query.ground_truth,
        query.metric,
        query.choices,
    )
    performance = rule_score
    extra_fields["evaluation_mode"] = "rule_only"
    extra_fields["evaluation_method"] = "rule"
    extra_fields["rule_score"] = rule_score
    extra_fields["final_score"] = performance

    if eval_config is not None:
        extra_fields["evaluation_mode"] = eval_config.mode
        extra_fields["judge_rubric_version"] = eval_config.rubric_version
        if eval_config.mode == "judge_review" and rule_score == 1.0:
            extra_fields["judge_reason"] = "rule_score=1"

        judge_score: Optional[float] = None
        should_judge = query.ground_truth is not None and _should_use_judge(
            eval_config,
            query.metric,
            rule_score,
        )

        if should_judge:
            if eval_client is None:
                eval_client = eval_config.get_client()
            judge_score, judge_meta = _evaluate_with_judge(
                eval_client=eval_client,
                eval_config=eval_config,
                query=query,
                response_text=response_text,
                judge_semaphore=judge_semaphore,
            )
            extra_fields.update(judge_meta)
            if judge_score is None and eval_config.hard_fail:
                raise RuntimeError(
                    extra_fields.get("judge_error", "judge evaluation failed")
                )

        final_score, fusion_strategy = _fuse_scores(
            eval_config,
            rule_score,
            judge_score,
        )
        performance = final_score
        extra_fields["final_score"] = final_score
        extra_fields["fusion_strategy"] = fusion_strategy

        if judge_score is not None:
            extra_fields["judge_score"] = judge_score
            if eval_config.mode == "judge_only":
                extra_fields["evaluation_method"] = "judge"
            elif eval_config.mode == "judge_review":
                extra_fields["evaluation_method"] = "judge_review"
            else:
                extra_fields["evaluation_method"] = "hybrid"

    return performance, extra_fields


def _evaluate_existing_record(
    record: ExistingResponseRecord,
    eval_config: Optional[EvalConfig],
    eval_client: Optional[OpenAI],
    judge_semaphore: Optional[threading.Semaphore],
) -> BenchmarkResult:
    performance, extra_fields = score_existing_response(
        query=record.query_record,
        response_text=record.response,
        eval_config=eval_config,
        eval_client=eval_client,
        judge_semaphore=judge_semaphore,
    )

    return BenchmarkResult(
        query=record.query_record.query,
        model_name=record.model_name,
        response=record.response,
        performance=performance,
        response_time=record.response_time,
        ground_truth=record.query_record.ground_truth,
        task_name=record.query_record.task_name,
        metric=record.query_record.metric,
        embedding_id=record.query_record.embedding_id,
        choices=record.query_record.choices,
        category=record.query_record.category,
        extra_fields=extra_fields,
    )


def _build_timeout_placeholder_for_existing_record(
    record: ExistingResponseRecord,
    evaluation_mode: str,
) -> BenchmarkResult:
    query = record.query_record
    rule_score = evaluate_response(
        record.response,
        query.ground_truth,
        query.metric,
        query.choices,
    )
    extra_fields = dict(query.extra_fields)
    extra_fields["evaluation_mode"] = evaluation_mode
    extra_fields["evaluation_method"] = "rule"
    extra_fields["rule_score"] = rule_score
    extra_fields["final_score"] = rule_score
    if evaluation_mode == "judge_review" and rule_score == 1.0:
        extra_fields["judge_reason"] = "rule_score=1"
    else:
        extra_fields["judge_reason"] = JUDGE_TIMEOUT_PLACEHOLDER_REASON

    return BenchmarkResult(
        query=query.query,
        model_name=record.model_name,
        response=record.response,
        performance=rule_score,
        response_time=record.response_time,
        ground_truth=query.ground_truth,
        task_name=query.task_name,
        metric=query.metric,
        embedding_id=query.embedding_id,
        choices=query.choices,
        category=query.category,
        extra_fields=extra_fields,
    )


def run_evaluation_only(
    records: List[ExistingResponseRecord],
    concurrency: int = 4,
    progress: bool = True,
    on_progress=None,
    eval_config: Optional[EvalConfig] = None,
    output_path: Optional[Path] = None,
    resume: bool = False,
) -> List[BenchmarkResult]:
    all_records_with_keys = []
    input_responses_by_key: Dict[str, str] = {}
    for record in records:
        key = _build_resume_key(
            query=record.query_record.query,
            model_name=record.model_name,
            task_name=record.query_record.task_name,
            metric=record.query_record.metric,
            ground_truth=record.query_record.ground_truth,
            choices=record.query_record.choices,
            category=record.query_record.category,
        )
        all_records_with_keys.append((record, key))
        input_responses_by_key[key] = record.response

    completed_resume_keys: Set[str] = set()
    if resume:
        retry_on_empty_output_with_input_response = (
            eval_config is not None and eval_config.mode == "judge_review"
        )
        completed_resume_keys = _load_completed_resume_keys(
            output_path,
            input_responses_by_key=input_responses_by_key,
            retry_on_empty_output_with_input_response=(
                retry_on_empty_output_with_input_response
            ),
        )
        _compact_resume_output(
            output_path,
            input_responses_by_key=input_responses_by_key,
            retry_on_empty_output_with_input_response=(
                retry_on_empty_output_with_input_response
            ),
        )

    pending_records = [
        record
        for record, key in all_records_with_keys
        if not resume or key not in completed_resume_keys
    ]

    already_completed = len(all_records_with_keys) - len(pending_records)
    total_tasks = len(pending_records)
    print(f"\nEvaluate-only mode: scoring {total_tasks} existing responses")
    if resume:
        print(
            f"Resume mode: {already_completed} responses already completed, {total_tasks} pending"
        )
    print(f"Concurrency: {concurrency}")
    print("No model inference requests will be sent in this mode.")
    print()

    results: List[Optional[BenchmarkResult]] = [None] * total_tasks
    completed = 0
    failed = 0
    timed_out = 0
    written = 0
    next_to_write = 0
    finished: List[bool] = [False] * total_tasks
    task_deadlines: List[Optional[float]] = [None] * total_tasks
    judge_semaphore: Optional[threading.Semaphore] = None
    eval_client: Optional[OpenAI] = None
    if eval_config is not None and eval_config.mode != "rule_only":
        judge_semaphore = threading.Semaphore(max(1, eval_config.concurrency))
        eval_client = eval_config.get_client()
    evaluation_mode = eval_config.mode if eval_config is not None else "rule_only"
    ordered_write_timeout_seconds = (
        eval_config.timeout_seconds
        if eval_config is not None
        else DEFAULT_EVAL_TIMEOUT_SECONDS
    )

    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "a" if resume else "w", encoding="utf-8")

    try:
        if total_tasks == 0:
            print("No pending responses. Nothing to evaluate.")
            return []

        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures: Dict[Any, int] = {}
        future_by_idx: Dict[int, Any] = {}

        for idx, record in enumerate(pending_records):
            future = executor.submit(
                _evaluate_existing_record,
                record,
                eval_config,
                eval_client,
                judge_semaphore,
            )
            futures[future] = idx
            future_by_idx[idx] = future

        pending_futures = set(futures.keys())
        pbar = tqdm(total=total_tasks, desc="Evaluating") if progress else None

        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                idx = futures[future]
                if finished[idx]:
                    continue
                try:
                    results[idx] = future.result()
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"\nError evaluating existing response: {e}")
                finally:
                    finished[idx] = True
                    if pbar is not None:
                        pbar.update(1)

            if output_file is not None:
                now = time.time()
                while next_to_write < total_tasks:
                    if finished[next_to_write]:
                        task_deadlines[next_to_write] = None
                        pending_result = results[next_to_write]
                        if pending_result is None:
                            pending_result = _build_timeout_placeholder_for_existing_record(
                                pending_records[next_to_write],
                                evaluation_mode,
                            )
                            results[next_to_write] = pending_result
                        output_file.write(json.dumps(pending_result.to_jsonl_dict()) + "\n")
                        output_file.flush()
                        written += 1
                        next_to_write += 1
                        continue

                    if task_deadlines[next_to_write] is None:
                        task_deadlines[next_to_write] = (
                            now
                            + ordered_write_timeout_seconds
                            + ORDERED_WRITE_STALL_GRACE_SECONDS
                        )
                        break

                    if now >= task_deadlines[next_to_write]:
                        timeout_result = _build_timeout_placeholder_for_existing_record(
                            pending_records[next_to_write],
                            evaluation_mode,
                        )
                        results[next_to_write] = timeout_result
                        finished[next_to_write] = True
                        timed_out += 1
                        timeout_future = future_by_idx.get(next_to_write)
                        if timeout_future is not None:
                            pending_futures.discard(timeout_future)
                        if pbar is not None:
                            pbar.update(1)
                        output_file.write(json.dumps(timeout_result.to_jsonl_dict()) + "\n")
                        output_file.flush()
                        written += 1
                        next_to_write += 1
                        continue

                    break

            if on_progress and total_tasks > 0:
                pct = int((completed + failed + timed_out) * 100 / total_tasks)
                on_progress(
                    pct,
                    "Evaluating",
                    f"{completed + failed + timed_out}/{total_tasks} responses processed ({failed} errors, {timed_out} timeouts)",
                )

        if pbar is not None:
            pbar.close()
    finally:
        if "executor" in locals():
            executor.shutdown(wait=False, cancel_futures=True)
        if output_file is not None:
            output_file.close()

    print(
        f"\nCompleted: {completed}/{total_tasks} ({failed} errors, {timed_out} timeouts)"
    )
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


def _build_eval_config(args: argparse.Namespace) -> Optional[EvalConfig]:
    if args.eval_mode == "rule_only":
        if args.eval_model:
            print(
                "Warning: --eval-model provided but --eval-mode=rule_only; judge settings are ignored"
            )
        return None

    if not args.eval_model:
        print(
            "Error: --eval-model is required when --eval-mode is "
            "judge_only, hybrid, or judge_review"
        )
        sys.exit(1)
    if args.eval_trigger == "uncertain" and (
        args.eval_uncertain_low >= args.eval_uncertain_high
    ):
        print("Error: --eval-uncertain-low must be < --eval-uncertain-high")
        sys.exit(1)
    if args.eval_fusion == "weighted" and not (0.0 <= args.eval_alpha <= 1.0):
        print("Error: --eval-alpha must be within [0, 1]")
        sys.exit(1)

    return EvalConfig(
        model=args.eval_model,
        endpoint=args.eval_endpoint or args.endpoint,
        api_key=args.eval_api_key or args.api_key,
        temperature=args.eval_temperature,
        timeout_seconds=max(1, args.eval_timeout_seconds),
        concurrency=max(1, args.concurrency),
        max_retries=max(0, args.eval_max_retries),
        hard_fail=args.eval_hard_fail,
        mode=args.eval_mode,
        trigger=args.eval_trigger,
        fusion=args.eval_fusion,
        alpha=args.eval_alpha,
        uncertain_low=args.eval_uncertain_low,
        uncertain_high=args.eval_uncertain_high,
        rubric_version=args.eval_rubric_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing response JSONL without sending benchmark inference requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_eval_only.py --input benchmark_output.jsonl --output rescored.jsonl
  python benchmark_eval_only.py --input benchmark_output.jsonl --output judge_review.jsonl --eval-mode judge_review --eval-model gpt-4o-mini --eval-endpoint https://api.openai.com/v1
  python benchmark_eval_only.py --input benchmark_output.jsonl --resume
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSONL file whose rows already contain response values.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_eval_output.jsonl",
        help="Output file path (must be different from --input)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output by skipping already completed entries",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"),
        help="Fallback endpoint for judge settings when --eval-endpoint is omitted",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        help="Fallback API key for judge settings when --eval-api-key is omitted",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Global concurrency limit for evaluation workers and judge requests",
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
        help="Limit number of response rows to process",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="",
        help="Judge model name for optional LLM-as-a-judge scoring",
    )
    parser.add_argument(
        "--eval-endpoint",
        type=str,
        default="",
        help="Judge API endpoint",
    )
    parser.add_argument(
        "--eval-api-key",
        type=str,
        default=os.environ.get("EVAL_API_KEY", ""),
        help="Judge API key (default: EVAL_API_KEY env var)",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Judge generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="rule_only",
        choices=["rule_only", "judge_only", "hybrid", "judge_review"],
        help="Evaluation mode for existing responses",
    )
    parser.add_argument(
        "--eval-trigger",
        type=str,
        default="by-metric",
        choices=["always", "uncertain", "by-metric"],
        help="When to invoke judge in hybrid mode",
    )
    parser.add_argument(
        "--eval-fusion",
        type=str,
        default="weighted",
        choices=["weighted", "rule", "judge", "max", "min"],
        help="Fusion strategy for hybrid mode",
    )
    parser.add_argument(
        "--eval-alpha",
        type=float,
        default=0.5,
        help="Weighted fusion alpha for rule score",
    )
    parser.add_argument(
        "--eval-uncertain-low",
        type=float,
        default=0.4,
        help="Lower bound for uncertain trigger",
    )
    parser.add_argument(
        "--eval-uncertain-high",
        type=float,
        default=0.8,
        help="Upper bound for uncertain trigger",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        help="Judge request timeout in seconds",
    )
    parser.add_argument(
        "--eval-max-retries",
        type=int,
        default=1,
        help="Judge request retries after first failure",
    )
    parser.add_argument(
        "--eval-hard-fail",
        action="store_true",
        help="Fail evaluation if judge fails instead of falling back to rule score",
    )
    parser.add_argument(
        "--eval-rubric-version",
        type=str,
        default="v1",
        help="Judge rubric version tag written to output metadata",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        print("Error: --output must be a new file and cannot be the same as --input")
        sys.exit(1)

    try:
        if not detect_evaluate_only_mode(input_path):
            print(
                "Error: input rows do not contain response values. "
                "Use benchmark_get_response.py for query-only benchmarking."
            )
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    records = load_response_records(input_path)
    if not records:
        print("Error: No valid response records loaded from file")
        sys.exit(1)

    if args.limit and args.limit > 0:
        original_count = len(records)
        records = records[: args.limit]
        print(f"Limited to {len(records)} records (from {original_count})")

    eval_config = _build_eval_config(args)

    results = run_evaluation_only(
        records=records,
        concurrency=args.concurrency,
        progress=not args.no_progress,
        eval_config=eval_config,
        output_path=output_path,
        resume=args.resume,
    )
    print_summary(results)


if __name__ == "__main__":
    main()