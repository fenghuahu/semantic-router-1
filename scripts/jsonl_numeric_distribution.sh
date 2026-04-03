#!/bin/bash

set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage:
  ./scripts/jsonl_numeric_distribution.sh <jsonl-file> <field> [options]

Description:
  Read numeric values from a JSONL file and print summary stats, bucket
  percentages, and a text histogram.

Arguments:
  <jsonl-file>    Path to the input JSONL file
  <field>         Field name or jq expression, for example:
                  performance
                  score.value
                  .performance

Options:
  --bins N                Number of buckets (default: 10)
  --min VALUE             Histogram lower bound (default: 0)
  --max VALUE             Histogram upper bound (default: 1)
    --split-exact VALUE     Put an exact numeric value in its own bucket
                                                    Can be repeated, e.g. --split-exact 0 --split-exact 1.0
  --no-histogram          Skip text bars and print counts only
  -h, --help              Show this help

Examples:
  ./scripts/jsonl_numeric_distribution.sh data.jsonl performance
  ./scripts/jsonl_numeric_distribution.sh data.jsonl performance --split-exact 1.0
    ./scripts/jsonl_numeric_distribution.sh data.jsonl performance --split-exact 0 --split-exact 1.0
  ./scripts/jsonl_numeric_distribution.sh data.jsonl score.value --bins 20 --min 0 --max 100
EOF
}

if [[ $# -lt 2 ]]; then
    show_usage
    exit 1
fi

jsonl_file="$1"
field_input="$2"
shift 2

bins=10
min_value=0
max_value=1
split_exact_values=()
show_histogram=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bins)
            bins="$2"
            shift 2
            ;;
        --min)
            min_value="$2"
            shift 2
            ;;
        --max)
            max_value="$2"
            shift 2
            ;;
        --split-exact)
            split_exact_values+=("$2")
            shift 2
            ;;
        --no-histogram)
            show_histogram=0
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$jsonl_file" ]]; then
    echo "Error: file not found: $jsonl_file" >&2
    exit 1
fi

if ! [[ "$bins" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --bins must be a positive integer" >&2
    exit 1
fi

if ! awk -v min="$min_value" -v max="$max_value" 'BEGIN { exit !(min < max) }'; then
    echo "Error: --min must be smaller than --max" >&2
    exit 1
fi

if [[ "$field_input" == .* ]]; then
    jq_field="$field_input"
else
    jq_field=".${field_input}"
fi

jq_filter="select(${jq_field} != null) | ${jq_field}"

split_exact_csv=""
if [[ ${#split_exact_values[@]} -gt 0 ]]; then
    split_exact_csv="$(IFS=,; echo "${split_exact_values[*]}")"
fi

jq -re "$jq_filter" "$jsonl_file" | awk \
    -v bins="$bins" \
    -v min_value="$min_value" \
    -v max_value="$max_value" \
    -v split_exact_csv="$split_exact_csv" \
    -v show_histogram="$show_histogram" \
    -v field_label="$field_input" '
function abs_value(x) {
    return x < 0 ? -x : x
}

function almost_equal(a, b) {
    return abs_value(a - b) < 1e-12
}

function sort_array(values, count,    i, j, key) {
    for (i = 1; i < count; i++) {
        key = values[i]
        j = i - 1
        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j]
            j--
        }
        values[j + 1] = key
    }
}

function quantile(values, count, q, q_index) {
    if (count == 0) {
        return 0
    }
    q_index = int(q * (count - 1))
    return values[q_index]
}

BEGIN {
    total = 0
    in_range = 0
    lower_outliers = 0
    upper_outliers = 0
    use_split = (split_exact_csv != "")

    split_values_count = 0
    if (use_split) {
        split_values_count = split(split_exact_csv, split_tokens, /,/)
        for (i = 1; i <= split_values_count; i++) {
            token = split_tokens[i]
            if (token == "") {
                continue
            }
            split_order[++split_order_len] = token
            split_value_num[token] = token + 0
            split_count[token] = 0
        }
    }
}

{
    value = $1 + 0
    total++

    if (use_split) {
        for (s = 1; s <= split_order_len; s++) {
            split_key = split_order[s]
            if (almost_equal(value, split_value_num[split_key])) {
                split_count[split_key]++
                raw_values[total - 1] = value
                next_record = 1
                break
            }
        }
        if (next_record) {
            next_record = 0
            next
        }
    }

    if (value < min_value) {
        lower_outliers++
        raw_values[total - 1] = value
        next
    }

    if (value > max_value) {
        upper_outliers++
        raw_values[total - 1] = value
        next
    }

    bucket = int((value - min_value) / (max_value - min_value) * bins)
    if (bucket == bins) {
        bucket = bins - 1
    }
    counts[bucket]++
    in_range++
    raw_values[total - 1] = value
}

END {
    if (total == 0) {
        print "No numeric values found."
        exit 0
    }

    for (i = 0; i < total; i++) {
        sorted_values[i] = raw_values[i]
    }
    sort_array(sorted_values, total)

    sum = 0
    min_seen = sorted_values[0]
    max_seen = sorted_values[total - 1]
    for (i = 0; i < total; i++) {
        sum += sorted_values[i]
    }
    mean = sum / total

    max_bucket_count = 0
    for (i = 0; i < bins; i++) {
        current = counts[i] + 0
        if (current > max_bucket_count) {
            max_bucket_count = current
        }
    }
    if (use_split) {
        for (s = 1; s <= split_order_len; s++) {
            split_key = split_order[s]
            current = split_count[split_key] + 0
            if (current > max_bucket_count) {
                max_bucket_count = current
            }
        }
    }

    printf "Field: %s\n", field_label
    printf "Count: %d\n", total
    printf "Min/Mean/Median/P90/Max: %.6f / %.6f / %.6f / %.6f / %.6f\n", min_seen, mean, quantile(sorted_values, total, 0.5), quantile(sorted_values, total, 0.9), max_seen
    printf "Range config: [%.6f, %.6f]\n", min_value, max_value
    printf "Out-of-range (<min, >max): %d, %d\n\n", lower_outliers, upper_outliers

    print "Bucket distribution:"
    for (i = 0; i < bins; i++) {
        left = min_value + i * (max_value - min_value) / bins
        right = min_value + (i + 1) * (max_value - min_value) / bins
        count = counts[i] + 0
        pct = 100 * count / total

        if (show_histogram && max_bucket_count > 0) {
            bar_len = int(count * 40 / max_bucket_count)
            bar = ""
            for (j = 0; j < bar_len; j++) {
                bar = bar "#"
            }
            printf "[%.6f, %.6f)%s %6d %7.2f%% %s\n", left, right, (i == bins - 1 ? "*" : " "), count, pct, bar
        } else {
            printf "[%.6f, %.6f)%s %6d %7.2f%%\n", left, right, (i == bins - 1 ? "*" : " "), count, pct
        }
    }

    if (use_split) {
        for (s = 1; s <= split_order_len; s++) {
            split_key = split_order[s]
            split_value = split_count[split_key] + 0
            pct = 100 * split_value / total
            if (show_histogram && max_bucket_count > 0) {
                bar_len = int(split_value * 40 / max_bucket_count)
                bar = ""
                for (j = 0; j < bar_len; j++) {
                    bar = bar "#"
                }
                printf "[%s]%*s %6d %7.2f%% %s\n", split_key, 14 - length(split_key), " ", split_value, pct, bar
            } else {
                printf "[%s]%*s %6d %7.2f%%\n", split_key, 14 - length(split_key), " ", split_value, pct
            }
        }
    }

    print "\n* The last interval is right-open in the display. Use --split-exact when a boundary value such as 1.0 should appear separately."
}'