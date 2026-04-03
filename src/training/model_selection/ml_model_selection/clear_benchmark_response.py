#!/usr/bin/env python3
"""Reset benchmark response fields in a JSONL file.

Reads an input JSONL file line by line, optionally keeps only rows matching a
specific `model_name`, then sets:
- `response` to null
- `response_time` to 0
- `performance` to 0.0

Then writes the updated records to a new JSONL file.

Usage:
    python clear_benchmark_response.py --input input.jsonl --output output.jsonl

    python clear_benchmark_response.py \
        --input input.jsonl \
        --output output.jsonl \
        --model-name llama-3.2-1b

Arguments:
    --input
        Path to the source JSONL file.

    --output
        Path to the destination JSONL file.
        Must be different from `--input`.

    --model-name
        Optional. When provided, only rows whose `model_name` equals this value
        are written to the output file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def reset_jsonl_fields(
    input_path: Path,
    output_path: Path,
    model_name: Optional[str] = None,
) -> tuple[int, int, int]:
    """Reset response-related fields for each JSON object in a JSONL file."""
    processed = 0
    skipped = 0
    filtered_out = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping invalid JSON at line {line_num}: {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            if not isinstance(record, dict):
                print(
                    f"Warning: skipping non-object JSON at line {line_num}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            if model_name and str(record.get("model_name") or "") != model_name:
                filtered_out += 1
                continue

            record["response"] = None
            record["response_time"] = 0
            record["performance"] = 0.0

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    return processed, skipped, filtered_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset response, response_time, and performance fields in a JSONL file."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--model-name",
        default="",
        help="Only process rows whose model_name matches this value",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path == output_path:
        print("Error: output file must be different from input file", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed, skipped, filtered_out = reset_jsonl_fields(
        input_path,
        output_path,
        model_name=args.model_name or None,
    )

    summary = f"Done. Wrote {processed} records to {output_path}"
    extras = []
    if skipped:
        extras.append(f"{skipped} skipped")
    if filtered_out:
        extras.append(f"{filtered_out} filtered out")
    if extras:
        summary += " (" + ", ".join(extras) + ")"

    print(summary)


if __name__ == "__main__":
    main()
