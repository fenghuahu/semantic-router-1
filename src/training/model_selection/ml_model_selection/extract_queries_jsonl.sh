#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <input.jsonl> [output.jsonl]" >&2
    echo "Extract unique {query, ground_truth, category} rows from a JSONL file." >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

input_file=$1
output_file=${2:-queries.jsonl}

if [[ ! -f "$input_file" ]]; then
    echo "Error: input file not found: $input_file" >&2
    exit 1
fi

if [[ "$input_file" == "$output_file" ]]; then
    echo "Error: output file must be different from input file" >&2
    exit 1
fi

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

jq -cs '
  reduce .[] as $row (
    {out: [], seen: {}};
    if ($row.query == null or $row.ground_truth == null) then
      error("each row must contain query and ground_truth")
    else
      ($row | {
        query: .query,
        ground_truth: .ground_truth,
        category: (.category // null)
      }) as $item
      | if .seen[$item.query] == null then
          .seen[$item.query] = $item | .out += [$item]
        elif .seen[$item.query] != $item then
          error("conflicting ground_truth/category for query: " + ($item.query | tostring))
        else
          .
        end
    end
  )
  | .out[]
' "$input_file" > "$tmp_file"

mv "$tmp_file" "$output_file"
trap - EXIT

echo "Wrote $output_file"