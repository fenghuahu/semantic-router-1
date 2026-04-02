#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def parse_field_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []

    fields = []
    for field in raw_value.split(","):
        normalized = field.strip()
        if normalized and normalized not in fields:
            fields.append(normalized)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unique rows from a JSONL file, deduplicated by query while "
            "preserving input order."
        )
    )
    parser.add_argument("input_jsonl", help="Path to the input JSONL file")
    parser.add_argument(
        "output_jsonl",
        nargs="?",
        default="queries.jsonl",
        help="Path to the output JSONL file (default: queries.jsonl)",
    )
    parser.add_argument(
        "--keep",
        help=(
            "Comma-separated list of fields to retain. Cannot be used with --drop. "
            "At least one of --keep or --drop is required."
        ),
    )
    parser.add_argument(
        "--drop",
        help=(
            "Comma-separated list of fields to remove while keeping all other input "
            "fields. Cannot be used with --keep."
        ),
    )
    return parser


def select_output_fields(
    row: dict[str, object],
    keep_fields: list[str] | None,
    drop_fields: set[str] | None,
) -> list[str]:
    if keep_fields is not None:
        return keep_fields

    if drop_fields is not None:
        return [field for field in row.keys() if field not in drop_fields]

    raise ValueError("either keep_fields or drop_fields must be provided")


def load_deduped_rows(
    input_path: Path,
    keep_fields: list[str] | None,
    drop_fields: set[str] | None,
) -> list[dict[str, object]]:
    seen_by_query: dict[str, dict[str, object]] = {}
    deduped_rows: list[dict[str, object]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {input_path} line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"expected JSON object in {input_path} line {line_number}"
                )

            query = row.get("query")
            if query is None:
                raise ValueError(
                    f"missing required field 'query' in {input_path} line {line_number}"
                )

            output_fields = select_output_fields(row, keep_fields, drop_fields)
            item = {field: row.get(field) for field in output_fields}
            existing = seen_by_query.get(query)

            if existing is None:
                seen_by_query[query] = item
                deduped_rows.append(item)
                continue

            if existing != item:
                raise ValueError(
                    "conflicting retained fields for query: "
                    f"{query}"
                )

    return deduped_rows


def write_jsonl(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")

    if input_path.resolve() == output_path.resolve():
        parser.error("output file must be different from input file")

    if args.keep and args.drop:
        parser.error("--keep and --drop cannot be used together")

    if not args.keep and not args.drop:
        parser.print_usage(sys.stderr)
        return 1

    keep_fields = parse_field_list(args.keep) if args.keep else None
    drop_fields = set(parse_field_list(args.drop)) if args.drop else None

    if keep_fields is not None and not keep_fields:
        parser.error("--keep must include at least one field")

    if keep_fields is not None and "query" not in keep_fields:
        parser.error("--keep must include query")

    if drop_fields is not None and "query" in drop_fields:
        parser.error("--drop cannot remove query")

    deduped_rows = load_deduped_rows(input_path, keep_fields, drop_fields)
    write_jsonl(output_path, deduped_rows)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())