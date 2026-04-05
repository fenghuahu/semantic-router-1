#!/usr/bin/env python3
"""
Benchmark script for ML model selection training data generation.

This script:
1. Takes input file (JSONL format) - can be queries or existing training data
2. Automatically extracts unique queries with their metadata (category, ground_truth, etc.)
3. Runs each query against multiple LLM endpoints
4. Measures performance (accuracy) and response_time
5. Outputs training-ready JSONL with category preserved

Input formats supported:
- Simple queries: {"query": "...", "ground_truth": "...", "category": "..."}
- Existing training data: {"query": "...", "model_name": "...", "performance": ..., "category": "..."}
  (Will extract unique queries and re-benchmark against YOUR models)

Usage:
    # Use existing training data file - extracts queries and benchmarks your models
    python benchmark.py --queries training_data.jsonl --model-config models.yaml

    # Simple: All models on same endpoint (e.g., vLLM or Ollama)
    python benchmark.py --queries queries.jsonl --models llama3.2:1b,mistral:7b

    # Different endpoints/auth per model: Use config file
    python benchmark.py --queries queries.jsonl --model-config models.yaml

    # Output to specific file
    python benchmark.py --queries queries.jsonl --models llama3.2:1b --output my_benchmark.jsonl

Config file format (models.yaml):
    models:
      - name: llama3.2:1b
        endpoint: http://localhost:11434/v1  # Ollama

      - name: llama3.2:3b
        endpoint: http://localhost:11434/v1

      - name: gpt-4
        endpoint: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}  # Environment variable

      - name: custom-model
        endpoint: https://custom.api.com/v1
        headers:
          Authorization: Bearer ${CUSTOM_TOKEN}
          X-Custom-Header: value
                enable_thinking: false
                enable_thinking_format: chat_template_kwargs  # or: direct

Output is training-ready (includes category if present in input):
    {"query": "...", "model_name": "...", "performance": 0.85, "response_time": 1.2, "category": "math"}

Then train with:
    python train.py --data-file benchmark_output.jsonl --output-dir models/
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


ENABLE_THINKING_FORMAT_DIRECT = "direct"
ENABLE_THINKING_FORMAT_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
ENABLE_THINKING_FORMAT_CHOICES = {
    ENABLE_THINKING_FORMAT_DIRECT,
    ENABLE_THINKING_FORMAT_CHAT_TEMPLATE_KWARGS,
}


def _normalize_enable_thinking_format(value: Optional[str], context: str) -> str:
    """Normalize and validate enable_thinking payload format."""
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
        """Create OpenAI client for this model."""
        # Resolve environment variables in api_key
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        # Resolve environment variables in headers
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


@dataclass
class EvalConfig:
    """Configuration for optional LLM-as-a-judge evaluation."""

    model: str
    endpoint: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout_seconds: int = 20
    concurrency: int = 2
    max_retries: int = 1
    hard_fail: bool = False
    mode: str = "rule_only"  # rule_only | judge_only | hybrid
    trigger: str = "by-metric"  # always | uncertain | by-metric
    fusion: str = "weighted"  # weighted | rule | judge | max | min
    alpha: float = 0.5
    uncertain_low: float = 0.4
    uncertain_high: float = 0.8
    rubric_version: str = "v1"

    def get_client(self) -> OpenAI:
        """Create OpenAI client for judge model."""
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        return OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
            timeout=float(self.timeout_seconds),
        )


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    """Load model configurations from YAML file."""
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
    """Create model configs from simple comma-separated list."""
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
    category: Optional[str] = None  # Domain category (e.g., "math", "physics")
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single query against a model."""

    query: str
    model_name: str
    response: str
    performance: float  # 0.0 - 1.0
    response_time: float  # seconds
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    category: Optional[str] = None  # Domain category (preserved from input)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Convert to JSONL-compatible dict (same format as benchmark_training_data.jsonl)."""
        result = {
            "query": self.query,
            "model_name": self.model_name,
            "response": self.response,
            "performance": self.performance,
            "response_time": self.response_time,
        }

        # Add optional fields
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

        # Add any extra fields from input
        result.update(self.extra_fields)

        return result


@dataclass
class ExistingResponseRecord:
    """A single existing response record for evaluate-only mode."""

    model_name: str
    response: str
    response_time: float
    query_record: QueryRecord


def format_concise_query(
    query: str, metric: Optional[str] = None, choices: Optional[str] = None
) -> str:
    """
    Format query with concise prompts to get shorter responses.
    Similar to Go benchmark runner's formatQueryForTask.
    """
    # Multiple choice questions
    # if choices or metric == "em_mc":
    if metric == "em_mc":
        return f"Answer with ONLY the letter of the correct choice (A, B, C, or D). Do not explain.\n\nQuestion: {query}\n\nChoices: {choices}"

    # Math problems
    if metric in ("MATH", "GSM8K"):
        return (
            f"{query}\n\nAnswer with ONLY the final number or expression. Be concise."
        )

    # Code generation
    if metric == "code_eval":
        return f"Write code to solve this problem. Output ONLY the code, no explanations:\n\n{query}"

    # commongen_coverage
    if metric == "commongen_coverage":
        return f"Given the following concepts, write a coherent and natural sentence that includes all the concepts:\n\nConcepts: {query}"

    # QA and general questions
    return f"Answer the following question concisely in one sentence:\n\n{query}"


def load_queries(file_path: Path, deduplicate: bool = True) -> List[QueryRecord]:
    """
    Load queries from JSONL file.

    Supports both formats:
    - Simple queries: {"query": "...", "ground_truth": "...", "category": "..."}
    - Full training data: {"query": "...", "model_name": "...", "performance": ..., "category": "..."}

    If deduplicate=True (default), extracts unique queries and preserves their metadata.
    This allows using existing training data files as input for benchmarking new models.

    Args:
        file_path: Path to JSONL file
        deduplicate: If True, return only unique queries (first occurrence wins for metadata)

    Returns:
        List of QueryRecord objects
    """
    seen_queries: Dict[str, QueryRecord] = {}  # query text -> record
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

                # Skip if we've seen this query and deduplication is enabled
                if deduplicate and query in seen_queries:
                    continue

                # Extract known fields
                record = QueryRecord(
                    query=query,
                    ground_truth=data.get("ground_truth"),
                    task_name=data.get("task_name"),
                    metric=data.get("metric"),
                    embedding_id=data.get("embedding_id"),
                    choices=data.get("choices"),
                    category=data.get("category"),  # Preserve category from input
                )

                # Store any extra fields (excluding benchmark-specific fields)
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

    # Print category distribution if categories exist
    categories = [r.category for r in records if r.category]
    if categories:
        from collections import Counter

        cat_counts = Counter(categories)
        print(f"Categories: {dict(cat_counts)}")

    return records


def detect_evaluate_only_mode(file_path: Path) -> bool:
    """Detect whether input JSONL contains existing responses for evaluate-only mode.

    Returns True when all query rows contain a non-null response field.
    Raises ValueError for mixed input (some rows have response, some do not).
    """

    query_lines: List[int] = []
    response_lines: List[int] = []
    missing_response_lines: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
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
    """Load existing response rows for evaluate-only mode."""

    records: List[ExistingResponseRecord] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
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


def evaluate_response(
    response: str,
    ground_truth: Optional[str],
    metric: Optional[str] = None,
    choices: Optional[str] = None,
) -> float:
    """
    Evaluate response against ground truth using metric-specific logic.

    Returns:
        Performance score between 0.0 and 1.0
    """
    if ground_truth is None:
        # No ground truth - return 0.5 as neutral score
        return 0.5

    response_lower = response.lower().strip()
    truth_lower = ground_truth.lower().strip()

    # Exact match (works for any metric)
    if response_lower == truth_lower:
        return 1.0

    # Metric-specific evaluation
    if metric == "em_mc" or choices:
        # Multiple choice - extract letter from response
        return _evaluate_multiple_choice(response, ground_truth, choices)

    elif metric == "GSM8K":
        # GSM8K - extract number after #### delimiter
        return _evaluate_gsm8k(response, ground_truth)

    elif metric == "MATH":
        # MATH - extract from \boxed{} with LaTeX normalization
        return _evaluate_math(response, ground_truth)

    elif metric == "f1_score":
        # F1 score based on word overlap
        return _evaluate_f1(response, ground_truth)

    elif metric == "code_eval":
        # Code evaluation - try to run assertions
        return _evaluate_code(response, ground_truth)

    elif metric == "commongen_coverage":
        # Check how many required words appear in response
        return _evaluate_commongen(response, ground_truth)

    else:
        # Default: CEM (Conditional Exact Match) - LLMRouter's default
        return _evaluate_cem(response, ground_truth)


def _evaluate_multiple_choice(
    response: str, ground_truth: str, choices: Optional[str]
) -> float:
    """
    Evaluate multiple choice questions by extracting answer letter.
    Aligned with LLMRouter's em_mc metric.
    """
    response_text = response.strip()
    truth_upper = ground_truth.upper().strip()

    # If ground truth is a single letter, look for it in response
    if len(truth_upper) == 1 and truth_upper in "ABCDEFGHIJ":
        # LLMRouter's approach: look for (A), (B), etc. pattern
        parenthesis_pattern = re.findall(r"\(\s*([a-zA-Z])\s*\)", response_text)
        if parenthesis_pattern:
            # Take the last match (usually the final answer)
            found_letter = parenthesis_pattern[-1].upper()
            return 1.0 if found_letter == truth_upper else 0.0

        # Additional patterns for various response styles
        patterns = [
            r"(?:answer(?:\s*is)?:?\s*)([A-J])\b",  # "answer is X"
            r"(?:it['\u2019]?s|is)\s+([A-J])\b",  # "it's X" or "is X"
            r"['\u2019]s\s+([A-J])\b",  # "'s X" pattern
            r"\b([A-J])\s+(?:because|since|as)",  # "X because..."
            r"(?:think|believe|choose)\s+([A-J])\b",  # "think X"
            r"\b([A-J])\s*[.)\]:]",  # Letter followed by punctuation
            r"^([A-J])[.)\]:\s]*$",  # Just the letter (with optional punctuation)
            r"\b([A-J])$",  # Ends with letter
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                found_letter = match.group(1).upper()
                # Skip "I" as a pronoun unless it's clearly an answer
                if found_letter == "I" and not re.match(
                    r"^I[.)\]:\s]*$", response_text.strip(), re.IGNORECASE
                ):
                    continue
                if found_letter == truth_upper:
                    return 1.0
                else:
                    return 0.0  # Wrong letter

        # Fallback: check if the letter appears standalone
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
    """
    Evaluate GSM8K math problems.
    Aligned with LLMRouter's gsm8k metric - splits on #### delimiter.
    """
    # Extract answer from ground truth (format: "explanation #### answer")
    if "####" in ground_truth:
        ground_truth_processed = ground_truth.split("####")[-1]
    else:
        ground_truth_processed = ground_truth

    # Clean the ground truth answer
    ground_truth_processed = (
        ground_truth_processed.replace(",", "")
        .replace("$", "")
        .replace(".", "")
        .strip()
    )

    # Extract numbers from response
    numbers = re.findall(r"(\-?[0-9\.\,]+)", response)
    if not numbers:
        return 0.0

    # Find the last valid number (usually the final answer)
    invalid_str = ["", "."]
    final_answer = None
    for answer in reversed(numbers):
        if answer not in invalid_str:
            final_answer = answer
            break

    if final_answer is None:
        return 0.0

    # Clean the predicted answer
    final_answer = (
        final_answer.replace(",", "").replace("$", "").replace(".", "").strip()
    )

    return 1.0 if final_answer == ground_truth_processed else 0.0


def _strip_latex_string(string: str) -> str:
    """
    Normalize LaTeX string for comparison.
    Aligned with LLMRouter's strip_string function.
    """
    # Remove linebreaks and spaces
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    # Normalize fractions
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degrees
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs and percentage
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Handle decimal points
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string

    # Remove "x = " or "k = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Remove spaces
    string = string.replace(" ", "")

    return string.strip()


def _last_boxed_string(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{} content from text.
    Aligned with LLMRouter's last_boxed_only_string function.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    # Find matching braces
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
    """Remove \\boxed{} wrapper and return content."""
    if "\\boxed{" in text:
        # Find the content inside \boxed{}
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
    elif "\\boxed " in text:
        return text.split("\\boxed ")[-1].split()[0]
    return text


def _evaluate_math(response: str, ground_truth: str) -> float:
    """
    Evaluate MATH problems by extracting \\boxed{} answers.
    Aligned with LLMRouter's math metric with LaTeX normalization.
    """
    # Extract ground truth from \boxed{} if present
    gt_boxed = _last_boxed_string(ground_truth)
    if gt_boxed:
        ground_truth_processed = _remove_boxed(gt_boxed)
    else:
        ground_truth_processed = ground_truth.strip()

    # Try to extract answer from response's \boxed{}
    try:
        response_boxed = _last_boxed_string(response)
        if response_boxed:
            response_answer = _remove_boxed(response_boxed)
            # Compare with LaTeX normalization
            if _strip_latex_string(response_answer) == _strip_latex_string(
                ground_truth_processed
            ):
                return 1.0
    except Exception:
        pass

    # Fallback: check if normalized ground truth appears in response
    gt_normalized = _strip_latex_string(ground_truth_processed)
    response_normalized = _strip_latex_string(response)

    if gt_normalized and gt_normalized in response_normalized:
        return 0.8

    # Try numeric comparison
    try:
        gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth_processed)
        resp_nums = re.findall(r"-?\d+\.?\d*", response)
        if gt_nums and resp_nums:
            if gt_nums[-1] in resp_nums:
                return 0.7
    except Exception:
        pass

    return 0.0


def _evaluate_f1(response: str, ground_truth: str) -> float:
    """Calculate F1 score based on word overlap."""
    # Clean punctuation and normalize
    import string

    def clean_words(text: str) -> set:
        # Remove punctuation and split
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, " ")
        return set(text.split())

    response_words = clean_words(response)
    truth_words = clean_words(ground_truth)

    if not truth_words:
        return 0.0

    # For short ground truths (1-2 words), check containment first
    if len(truth_words) <= 2:
        truth_text = ground_truth.lower()
        for p in string.punctuation:
            truth_text = truth_text.replace(p, "")
        if truth_text.strip() in response.lower():
            return 1.0

    # Remove common stopwords for better matching
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

    # If truth was only stopwords, use original
    if not truth_content:
        truth_content = truth_words
    if not response_content:
        response_content = response_words

    overlap = response_content & truth_content

    if not overlap:
        return 0.0

    precision = len(overlap) / len(response_content) if response_content else 0
    recall = len(overlap) / len(truth_content)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _evaluate_code(response: str, ground_truth: str, timeout: int = 5) -> float:
    """
    Evaluate code by trying to run assertions.
    Aligned with LLMRouter's evaluate_code with timeout protection.
    """
    import signal
    import sys

    # Try to extract code from response
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

    # Try to run assertions from ground truth
    try:
        # Ground truth is usually a list of assertions like:
        # "['assert func([1,2])==3', 'assert func([4])==4']"
        if ground_truth.startswith("[") and "assert" in ground_truth:
            assertions = eval(ground_truth)
            if isinstance(assertions, list):
                passed = 0
                total = len(assertions)

                # Timeout handler (Unix only)
                def timeout_handler(signum, frame):
                    raise TimeoutError("Code execution timed out")

                # Set timeout if signal.SIGALRM is available (Unix)
                alarm_supported = hasattr(signal, "SIGALRM")

                for assertion in assertions:
                    try:
                        if alarm_supported:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(timeout)

                        # Execute the code first, then the assertion
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

    # Fallback: check if function structure matches
    func_match = re.search(r"def\s+(\w+)", response)
    if func_match:
        func_name = func_match.group(1)
        if func_name in ground_truth.lower():
            return 0.5

    return 0.3  # Gave some code response


def _evaluate_commongen(response: str, ground_truth: str) -> float:
    """Evaluate commongen by checking word coverage."""
    # Ground truth is a comma-separated list of words
    required_words = set(w.strip().lower() for w in ground_truth.split(","))
    response_lower = response.lower()

    found = sum(1 for word in required_words if word in response_lower)

    return found / len(required_words) if required_words else 0.0


def _normalize_answer(text: str) -> str:
    """
    Normalize text for evaluation.
    Aligned with LLMRouter's normalize_answer function.
    """
    import string

    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Fix whitespace
    text = " ".join(text.split())

    return text


def _evaluate_cem(response: str, ground_truth: str) -> float:
    """
    CEM (Conditional Exact Match) evaluation - LLMRouter's default.
    Returns 1.0 if exact match OR ground_truth contained in response, else 0.0
    """
    norm_response = _normalize_answer(response)
    norm_gt = _normalize_answer(ground_truth)

    # Exact match or containment
    if norm_response == norm_gt or norm_gt in norm_response:
        return 1.0

    return 0.0


RULE_ONLY_METRICS = {"em_mc", "gsm8k", "math", "code_eval"}
HYBRID_ELIGIBLE_METRICS = {"cem", "f1_score", "commongen_coverage"}


def _normalize_metric(metric: Optional[str]) -> str:
    return (metric or "cem").strip().lower()


def _should_use_judge(
    eval_config: EvalConfig,
    metric: Optional[str],
    rule_score: float,
) -> bool:
    """Decide whether to invoke judge in hybrid mode."""
    if eval_config.mode == "judge_only":
        return True
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
    # Unknown metrics default to judge-eligible in hybrid mode.
    return True


def _build_judge_prompt(
    query: str,
    ground_truth: str,
    model_response: str,
    metric: Optional[str],
    rubric_version: str,
) -> str:
    """Build a strict JSON-only grading prompt for the judge model."""
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
    """Extract and parse the first balanced JSON object from text."""
    # First try direct parse.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _parse_judge_result(raw_text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """Parse judge output and normalize fields into metadata."""
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
    """Fuse rule and judge scores according to configured strategy."""
    if judge_score is None:
        return rule_score, "rule_fallback"

    if eval_config.mode == "judge_only":
        return judge_score, "judge_only"

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

    # weighted
    final_score = eval_config.alpha * rule_score + (1.0 - eval_config.alpha) * judge_score
    return final_score, "weighted"


def _evaluate_with_judge(
    eval_client: OpenAI,
    eval_config: EvalConfig,
    query: QueryRecord,
    response_text: str,
    judge_semaphore: Optional[threading.Semaphore],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Call judge model and return score plus metadata."""
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

    for attempt in range(attempts):
        try:
            if judge_semaphore is not None:
                with judge_semaphore:
                    judge_resp = eval_client.chat.completions.create(
                        model=eval_config.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=eval_config.temperature,
                    )
            else:
                judge_resp = eval_client.chat.completions.create(
                    model=eval_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=eval_config.temperature,
                )

            judge_text = judge_resp.choices[0].message.content or ""
            judge_score, parse_meta = _parse_judge_result(judge_text)
            metadata.update(parse_meta)
            return judge_score, metadata
        except Exception as e:
            last_error = str(e)
            if attempt + 1 >= attempts:
                break

    metadata["judge_error"] = f"judge_request_failed: {last_error}"
    return None, metadata


def score_existing_response(
    query: QueryRecord,
    response_text: str,
    eval_config: Optional[EvalConfig] = None,
    eval_client: Optional[OpenAI] = None,
    judge_semaphore: Optional[threading.Semaphore] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Score an existing response using rule and optional judge logic."""

    extra_fields = dict(query.extra_fields)

    rule_score = evaluate_response(
        response_text,
        query.ground_truth,
        query.metric,
        query.choices,
    )
    performance = rule_score

    # Default metadata values for explainability.
    extra_fields["evaluation_mode"] = "rule_only"
    extra_fields["evaluation_method"] = "rule"
    extra_fields["rule_score"] = rule_score
    extra_fields["final_score"] = performance

    if eval_config is not None:
        extra_fields["evaluation_mode"] = eval_config.mode
        extra_fields["judge_rubric_version"] = eval_config.rubric_version

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
            extra_fields["evaluation_method"] = (
                "judge" if eval_config.mode == "judge_only" else "hybrid"
            )

    return performance, extra_fields


def benchmark_query(
    model_config: ModelConfig,
    query: QueryRecord,
    concise: bool = False,
    eval_config: Optional[EvalConfig] = None,
    eval_client: Optional[OpenAI] = None,
    judge_semaphore: Optional[threading.Semaphore] = None,
) -> BenchmarkResult:
    """Benchmark a single query against a model."""

    client = model_config.get_client()
    start_time = time.time()

    # Format query with concise prompts if enabled
    # But skip concise for code_eval - code needs full response
    query_text = query.query
    # if concise and query.metric != "code_eval":
    if concise:
        query_text = format_concise_query(query.query, query.metric, query.choices)

    # Use higher max_tokens for code_eval (code needs more tokens)
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

    end_time = time.time()
    response_time = end_time - start_time

    # Evaluate performance
    extra_fields = dict(query.extra_fields)
    if success:
        performance, extra_fields = score_existing_response(
            query=query,
            response_text=response_text,
            eval_config=eval_config,
            eval_client=eval_client,
            judge_semaphore=judge_semaphore,
        )
    else:
        performance = 0.0
        extra_fields["evaluation_mode"] = "rule_only"
        extra_fields["evaluation_method"] = "rule"
        extra_fields["final_score"] = 0.0

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
        category=query.category,  # Preserve category from input
        extra_fields=extra_fields,
    )


def run_benchmark(
    queries: List[QueryRecord],
    model_configs: List[ModelConfig],
    concurrency: int = 4,
    progress: bool = True,
    concise: bool = False,
    on_progress=None,
    eval_config: Optional[EvalConfig] = None,
    output_path: Optional[Path] = None,
) -> List[BenchmarkResult]:
    """Run benchmark for all queries against all models.

    Args:
        on_progress: Optional callback(percent, step, message) called as tasks complete.
                     percent ranges 0-100 within the benchmark phase.
        output_path: Optional JSONL output path. When set, results are written
                     incrementally in deterministic task order.
    """

    results: List[Optional[BenchmarkResult]] = []

    # Create tasks: (query, model_config) pairs
    # Group by model to minimize model reloading (important for Ollama/local inference)
    tasks = [(q, m) for m in model_configs for q in queries]
    total_tasks = len(tasks)
    results = [None] * total_tasks

    # Group models by endpoint for display
    endpoints = set(m.endpoint for m in model_configs)
    model_names = [m.name for m in model_configs]

    print(
        f"\nBenchmarking {len(queries)} queries × {len(model_configs)} models = {total_tasks} requests"
    )
    print(f"Models: {', '.join(model_names)}")
    if len(endpoints) == 1:
        print(f"Endpoint: {list(endpoints)[0]}")
    else:
        print(f"Endpoints: {len(endpoints)} different endpoints")
        for m in model_configs:
            auth_info = "with API key" if m.api_key else "no auth"
            print(f"  - {m.name}: {m.endpoint} ({auth_info})")
    print(f"Concurrency: {concurrency}")
    print()

    completed = 0
    failed = 0
    written = 0
    next_to_write = 0
    finished: List[bool] = [False] * total_tasks
    judge_semaphore: Optional[threading.Semaphore] = None
    eval_client: Optional[OpenAI] = None
    if eval_config is not None and eval_config.mode != "rule_only":
        judge_semaphore = threading.Semaphore(max(1, eval_config.concurrency))
        eval_client = eval_config.get_client()

    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: Dict[Any, Tuple[int, QueryRecord, ModelConfig]] = {}

            for idx, (query, model_config) in enumerate(tasks):
                future = executor.submit(
                    benchmark_query,
                    model_config,
                    query,
                    concise,
                    eval_config,
                    eval_client,
                    judge_semaphore,
                )
                futures[future] = (idx, query, model_config)

            # Process results as they complete
            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=total_tasks, desc="Benchmarking")

            for future in iterator:
                idx, query, model_config = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    completed += 1

                    if result.performance == 0.0 and "Error" in result.response:
                        failed += 1

                    # Report per-task progress (scale 0-100 within this function)
                    if on_progress and total_tasks > 0:
                        pct = int(completed * 100 / total_tasks)
                        on_progress(
                            pct,
                            "Benchmarking",
                            f"{completed}/{total_tasks} queries completed ({failed} errors)",
                        )

                except Exception as e:
                    print(f"\nError processing {model_config.name}: {e}")
                    failed += 1
                finally:
                    finished[idx] = True
                    if output_file is not None:
                        while next_to_write < total_tasks and finished[next_to_write]:
                            pending_result = results[next_to_write]
                            if pending_result is not None:
                                output_file.write(
                                    json.dumps(pending_result.to_jsonl_dict()) + "\n"
                                )
                                output_file.flush()
                                written += 1
                            next_to_write += 1
    finally:
        if output_file is not None:
            output_file.close()

    print(f"\nCompleted: {completed}/{total_tasks} ({failed} errors)")
    if output_path is not None:
        print(f"Stream-saved {written} results to {output_path}")

    ordered_results = [r for r in results if r is not None]
    return ordered_results


def _evaluate_existing_record(
    record: ExistingResponseRecord,
    eval_config: Optional[EvalConfig],
    eval_client: Optional[OpenAI],
    judge_semaphore: Optional[threading.Semaphore],
) -> BenchmarkResult:
    """Evaluate one precomputed response and return benchmark-compatible output."""

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


def run_evaluation_only(
    records: List[ExistingResponseRecord],
    concurrency: int = 4,
    progress: bool = True,
    on_progress=None,
    eval_config: Optional[EvalConfig] = None,
    output_path: Optional[Path] = None,
) -> List[BenchmarkResult]:
    """Evaluate already-generated responses without querying any model endpoint."""

    total_tasks = len(records)
    print(f"\nEvaluate-only mode: scoring {total_tasks} existing responses")
    print(f"Concurrency: {concurrency}")
    print("No model inference requests will be sent in this mode.")
    print()

    results: List[Optional[BenchmarkResult]] = [None] * total_tasks
    completed = 0
    failed = 0
    written = 0
    next_to_write = 0
    finished: List[bool] = [False] * total_tasks
    judge_semaphore: Optional[threading.Semaphore] = None
    eval_client: Optional[OpenAI] = None
    if eval_config is not None and eval_config.mode != "rule_only":
        judge_semaphore = threading.Semaphore(max(1, eval_config.concurrency))
        eval_client = eval_config.get_client()

    output_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: Dict[Any, int] = {}
            for idx, record in enumerate(records):
                future = executor.submit(
                    _evaluate_existing_record,
                    record,
                    eval_config,
                    eval_client,
                    judge_semaphore,
                )
                futures[future] = idx

            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=total_tasks, desc="Evaluating")

            for future in iterator:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"\nError evaluating existing response: {e}")
                finally:
                    finished[idx] = True
                    if output_file is not None:
                        while next_to_write < total_tasks and finished[next_to_write]:
                            pending_result = results[next_to_write]
                            if pending_result is not None:
                                output_file.write(
                                    json.dumps(pending_result.to_jsonl_dict()) + "\n"
                                )
                                output_file.flush()
                                written += 1
                            next_to_write += 1

                if on_progress and total_tasks > 0:
                    pct = int((completed + failed) * 100 / total_tasks)
                    on_progress(
                        pct,
                        "Evaluating",
                        f"{completed + failed}/{total_tasks} responses processed ({failed} errors)",
                    )
    finally:
        if output_file is not None:
            output_file.close()

    print(f"\nCompleted: {completed}/{total_tasks} ({failed} errors)")
    if output_path is not None:
        print(f"Stream-saved {written} results to {output_path}")
    ordered_results = [r for r in results if r is not None]
    return ordered_results


def save_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSONL file."""

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_jsonl_dict()) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary."""

    print("\n" + "=" * 60)
    print("  Benchmark Summary")
    print("=" * 60)

    # Group by model
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

    for model, stats in sorted(model_stats.items()):
        avg_perf = stats["total_perf"] / stats["count"] if stats["count"] > 0 else 0
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        success_rate = (
            stats["successes"] / stats["count"] * 100 if stats["count"] > 0 else 0
        )

        print(
            f"{model:<25} {stats['count']:>8} {avg_perf:>10.3f} {avg_time:>9.2f}s {success_rate:>9.1f}%"
        )

    print("=" * 60)
    print()


def run_benchmark_pipeline(
    queries_path: str,
    models_yaml_path: str = None,
    models_list: List[str] = None,
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
    eval_config: Optional[EvalConfig] = None,
) -> List[BenchmarkResult]:
    """
    Run the full benchmark pipeline: load queries -> load models -> benchmark -> save.

    This is the shared entry point used by both the CLI (main()) and the
    HTTP service (server.py).

    Args:
        queries_path: Path to JSONL queries file.
        models_yaml_path: Path to YAML model config (mutually exclusive with models_list).
        models_list: List of model names sharing the same endpoint.
        endpoint: API endpoint when using models_list.
        api_key: API key when using models_list.
        output_path: Path to write output JSONL.
        concurrency: Number of concurrent requests.
        max_tokens: Max tokens in response.
        temperature: Temperature for generation.
        enable_thinking: Optional thinking mode flag to send in request body.
        enable_thinking_format: Optional payload format override for enable_thinking.
        concise: Use concise prompts.
        limit: Limit number of queries (0 = no limit).
        show_progress: Show progress bar (tqdm).
        on_progress: Optional callback(percent, step, message) for progress.
        eval_config: Optional configuration for judge-based evaluation.

    Returns:
        List of BenchmarkResult objects.
    """

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

    # Load queries
    qpath = Path(queries_path)
    if not qpath.exists():
        raise FileNotFoundError(f"Queries file not found: {qpath}")

    evaluate_only_mode = detect_evaluate_only_mode(qpath)

    if evaluate_only_mode:
        records = load_response_records(qpath)
        if not records:
            raise ValueError("No valid response records loaded from file")

        if limit and limit > 0:
            original_count = len(records)
            records = records[:limit]
            print(f"Limited to {len(records)} records (from {original_count})")

        progress(
            10,
            "Running evaluate-only",
            f"Scoring {len(records)} existing responses",
        )

        def eval_progress(pct, step, msg):
            scaled = 10 + int(pct * 80 / 100)
            progress(scaled, step, msg)

        results = run_evaluation_only(
            records=records,
            concurrency=concurrency,
            progress=show_progress,
            on_progress=eval_progress if on_progress else None,
            eval_config=eval_config,
            output_path=Path(output_path),
        )

        progress(92, "Finalizing output", f"Streamed {len(results)} results to {output_path}")
        progress(96, "Summary", "Generating summary")
        print_summary(results)
        return results

    queries = load_queries(qpath)
    if not queries:
        raise ValueError("No queries loaded from file")

    # Apply limit
    if limit and limit > 0:
        original_count = len(queries)
        queries = queries[:limit]
        print(f"Limited to {len(queries)} queries (from {original_count})")

    # Load model configs
    if models_yaml_path:
        config_path = Path(models_yaml_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        model_configs = load_model_configs(config_path)
        # Override max_tokens and temperature if specified
        for mc in model_configs:
            mc.max_tokens = max_tokens
            mc.temperature = temperature
            if enable_thinking is not None:
                mc.enable_thinking = enable_thinking
            if normalized_enable_thinking_format is not None:
                mc.enable_thinking_format = normalized_enable_thinking_format
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
                normalized_enable_thinking_format
                or ENABLE_THINKING_FORMAT_DIRECT
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

    # Scale on_progress from run_benchmark's 0-100 into pipeline's 10-90 range
    def benchmark_progress(pct, step, msg):
        # Map run_benchmark's 0-100% into the pipeline's 10-90% range
        scaled = 10 + int(pct * 80 / 100)
        progress(scaled, step, msg)

    # Run benchmark
    results = run_benchmark(
        queries=queries,
        model_configs=model_configs,
        concurrency=concurrency,
        progress=show_progress,
        concise=concise,
        on_progress=benchmark_progress if on_progress else None,
        eval_config=eval_config,
        output_path=Path(output_path),
    )

    progress(92, "Finalizing output", f"Streamed {len(results)} results to {output_path}")

    progress(96, "Summary", "Generating summary")

    # Print summary
    print_summary(results)

    return results


def main():
    def _parse_bool_arg(value: str) -> bool:
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(
            f"Invalid boolean value: {value}. Use true/false."
        )

    parser = argparse.ArgumentParser(
        description="Benchmark LLMs for ML model selection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing training data - extracts unique queries and benchmarks your models
  python benchmark.py --queries training_data_with_category.jsonl --model-config models.yaml

  # Ollama models (recommended config file approach)
  python benchmark.py --queries queries.jsonl --model-config ollama_models.yaml

  # Simple: All models on same endpoint (local vLLM)
  python benchmark.py --queries queries.jsonl --models llama-3.2-1b,mistral-7b

  # With custom endpoint and API key (OpenAI)
  python benchmark.py --queries queries.jsonl --models gpt-4 \\
      --endpoint https://api.openai.com/v1 --api-key $OPENAI_API_KEY

  # High concurrency for faster benchmarking
  python benchmark.py --queries queries.jsonl --models model1,model2 --concurrency 16

Config file format (models.yaml) - supports Ollama, vLLM, OpenAI, etc:
  models:
    - name: llama3.2:1b            # Ollama models (all same endpoint)
      endpoint: http://localhost:11434/v1
    - name: llama3.2:3b
      endpoint: http://localhost:11434/v1
    - name: mistral:7b
      endpoint: http://localhost:11434/v1
    - name: codellama:7b
      endpoint: http://localhost:11434/v1

    - name: gpt-4                   # OpenAI with API key
      endpoint: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}

    - name: custom-model            # Custom headers
      endpoint: https://custom.api.com/v1
      headers:
        Authorization: Bearer ${CUSTOM_TOKEN}
            enable_thinking: false
            enable_thinking_format: chat_template_kwargs  # or: direct

After benchmarking, train directly (category is preserved from input):
  python train.py --data-file benchmark_output.jsonl --output-dir models/ --device cuda
        """,
    )

    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to JSONL input file. If rows include response, runs evaluate-only mode; "
        "otherwise runs normal benchmark mode.",
    )

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names (all use same endpoint)",
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML config file with model definitions (supports different endpoints/auth per model)",
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
        help="Output file path (must be different from --queries; default: benchmark_output.jsonl)",
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
        help="Set enable_thinking value (true/false). "
        "When provided, --enable-thinking-format is required. "
        "When omitted, do not send this field.",
    )
    parser.add_argument(
        "--enable-thinking-format",
        type=str,
        choices=sorted(ENABLE_THINKING_FORMAT_CHOICES),
        default=None,
        help="Request payload format for --enable-thinking. "
        "direct -> extra_body.enable_thinking; "
        "chat_template_kwargs -> extra_body.chat_template_kwargs.enable_thinking. "
        "When set, this overrides per-model YAML config.",
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
        help="Judge API endpoint (defaults to --endpoint when omitted)",
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
        choices=["rule_only", "judge_only", "hybrid"],
        help="Evaluation mode: rule_only, judge_only, hybrid (default: rule_only)",
    )
    parser.add_argument(
        "--eval-trigger",
        type=str,
        default="by-metric",
        choices=["always", "uncertain", "by-metric"],
        help="When to invoke judge in hybrid mode (default: by-metric)",
    )
    parser.add_argument(
        "--eval-fusion",
        type=str,
        default="weighted",
        choices=["weighted", "rule", "judge", "max", "min"],
        help="Fusion strategy for hybrid mode (default: weighted)",
    )
    parser.add_argument(
        "--eval-alpha",
        type=float,
        default=0.5,
        help="Weighted fusion alpha for rule score (default: 0.5)",
    )
    parser.add_argument(
        "--eval-uncertain-low",
        type=float,
        default=0.4,
        help="Lower bound for uncertain trigger (default: 0.4)",
    )
    parser.add_argument(
        "--eval-uncertain-high",
        type=float,
        default=0.8,
        help="Upper bound for uncertain trigger (default: 0.8)",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=300,
        help="Judge request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=2,
        help="Judge concurrency limit (default: 2)",
    )
    parser.add_argument(
        "--eval-max-retries",
        type=int,
        default=1,
        help="Judge request retries after first failure (default: 1)",
    )
    parser.add_argument(
        "--eval-hard-fail",
        action="store_true",
        help="Fail benchmark if judge fails instead of falling back to rule score",
    )
    parser.add_argument(
        "--eval-rubric-version",
        type=str,
        default="v1",
        help="Judge rubric version tag written to output metadata",
    )

    args = parser.parse_args()

    if args.enable_thinking is not None and args.enable_thinking_format is None:
        parser.error(
            "--enable-thinking requires --enable-thinking-format "
            "(direct or chat_template_kwargs)"
        )

    # Safety guard: never overwrite the input file.
    queries_abs = Path(args.queries).expanduser().resolve()
    output_abs = Path(args.output).expanduser().resolve()
    if queries_abs == output_abs:
        print(
            "Error: --output must be a new file and cannot be the same as --queries"
        )
        sys.exit(1)

    evaluate_only_input = False
    try:
        if queries_abs.exists():
            evaluate_only_input = detect_evaluate_only_mode(queries_abs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not evaluate_only_input and not args.models and not args.model_config:
        print("Error: benchmark mode requires either --models or --model-config")
        sys.exit(1)

    if evaluate_only_input and (args.models or args.model_config):
        print(
            "Warning: evaluate-only input detected; --models/--model-config are ignored"
        )

    # Determine model list for simple mode
    models_list = None
    if args.models:
        models_list = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models_list and not evaluate_only_input:
            print("Error: No models specified")
            sys.exit(1)

    eval_config: Optional[EvalConfig] = None
    if args.eval_mode != "rule_only":
        if not args.eval_model:
            print("Error: --eval-model is required when --eval-mode is judge_only or hybrid")
            sys.exit(1)
        if args.eval_trigger == "uncertain" and (
            args.eval_uncertain_low >= args.eval_uncertain_high
        ):
            print("Error: --eval-uncertain-low must be < --eval-uncertain-high")
            sys.exit(1)
        if args.eval_fusion == "weighted" and not (0.0 <= args.eval_alpha <= 1.0):
            print("Error: --eval-alpha must be within [0, 1]")
            sys.exit(1)

        eval_config = EvalConfig(
            model=args.eval_model,
            endpoint=args.eval_endpoint or args.endpoint,
            api_key=args.eval_api_key or args.api_key,
            temperature=args.eval_temperature,
            timeout_seconds=max(1, args.eval_timeout_seconds),
            concurrency=max(1, args.eval_concurrency),
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
    elif args.eval_model:
        print("Warning: --eval-model provided but --eval-mode=rule_only; judge settings are ignored")

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
            eval_config=eval_config,
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
