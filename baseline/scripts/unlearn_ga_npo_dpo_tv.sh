#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a GA, NPO, DPO, or TV baseline on an Enron forget set.

Usage:
  bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh --model_dir DIR [OPTIONS]

Options:
  --algo NAME                 Algorithm (default: ga; case-insensitive).
  --model_dir DIR              Required: locally trained target model directory.
  --tokenizer_dir DIR          Tokenizer directory (default: model directory).
  --forget_rate RATE           0.2 or 0.5 (default: 0.2).
  --retain_data_file FILE      Required for *_gdr and *_klr.
  --positive_data_file FILE    DPO positive data (default: bundled IDK set for RATE).
  --alpha FLOAT               TV-only task-vector scale, non-negative (default: 1.0).
  --out_dir DIR                Output directory (default: REPO/ckpt/enron/ALGO/forget_RATE).
  --epochs N                  Training epochs (default: 10).
  --lr FLOAT                  Positive learning rate (default: 1e-5).
  --per_device_batch_size N    Batch size per device (default: 1).
  --max_len N                 Maximum sequence length, at least 2 (default: 2048).
  --dry-run                   Print the command without starting training.
  -h, --help                  Show this help message.

Algorithms:
  ga       ga_gdr       ga_klr
  npo      npo_gdr      npo_klr
  dpo      dpo_gdr      dpo_klr
  tv

Retain data is only accepted by *_gdr and *_klr; positive data is only
accepted by DPO methods. Data files must be .txt or .json (a list of strings
or objects with a "text" field). Custom DPO positive data must match the
forget set in sample count and question order.

TV fine-tunes on the forget set and subtracts the task vector scaled by alpha.
Training options control the fine-tuning step. The intermediate model is saved
to OUT_DIR_ft, and the final TV model is saved to OUT_DIR.

The forget set and training entry point are located relative to this script.
User-supplied relative paths are resolved from the current working directory.
Set PYTHON_BIN to select the interpreter in your prepared training environment.

Examples:
  bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
    --algo ga --model_dir ./models/target --forget_rate 0.2
  bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
    --algo npo_gdr --model_dir ./models/target --retain_data_file ./data/retain.txt
  bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
    --algo dpo --model_dir ./models/target --forget_rate 0.5
  bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
    --algo tv --model_dir ./models/target --forget_rate 0.2 --alpha 1.0
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

algo='ga'
model_dir=''
tokenizer_dir=''
forget_rate='0.2'
retain_data_file=''
positive_data_file=''
alpha=''
out_dir=''
epochs='10'
lr='1e-5'
per_device_batch_size='1'
max_len='2048'
dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo|--model_dir|--tokenizer_dir|--forget_rate|--retain_data_file|--positive_data_file|--alpha|--out_dir|--epochs|--lr|--per_device_batch_size|--max_len)
            [[ $# -ge 2 ]] || fail "Missing value for $1."
            [[ -n "$2" && "$2" != --* ]] || fail "Missing value for $1."
            printf -v "${1#--}" '%s' "$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1. Use --help for usage."
            ;;
    esac
done

[[ -n "$model_dir" ]] || fail '--model_dir is required. Use --help for usage.'

algo="$(printf '%s' "$algo" | tr '[:upper:]' '[:lower:]')"
case "$algo" in
    ga|ga_gdr|ga_klr|npo|npo_gdr|npo_klr|dpo|dpo_gdr|dpo_klr|tv) ;;
    *) fail "Unknown algorithm: $algo. Use --help for supported algorithms." ;;
esac

needs_retain=false
case "$algo" in
    *_gdr|*_klr)
        needs_retain=true
        [[ -n "$retain_data_file" ]] || fail "--retain_data_file is required for $algo."
        ;;
    *)
        [[ -z "$retain_data_file" ]] || fail "--retain_data_file is only used by *_gdr and *_klr, not $algo."
        ;;
esac

needs_positive=false
case "$algo" in
    dpo|dpo_*) needs_positive=true ;;
    *) [[ -z "$positive_data_file" ]] || fail "--positive_data_file is only used by DPO methods, not $algo." ;;
esac

case "$forget_rate" in
    0.2)  suffix='02' ;;
    0.5)  suffix='05' ;;
    *) fail '--forget_rate must be 0.2 or 0.5.' ;;
esac

[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || fail '--epochs must be a positive integer.'
[[ "$per_device_batch_size" =~ ^[1-9][0-9]*$ ]] || fail '--per_device_batch_size must be a positive integer.'
[[ "$max_len" =~ ^[1-9][0-9]*$ && "$max_len" != 1 ]] || fail '--max_len must be an integer of at least 2.'
number_pattern='^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
[[ "$lr" =~ $number_pattern && "${lr%%[eE]*}" =~ [1-9] ]] || fail '--lr must be a positive number, e.g. 1e-5.'

if [[ "$algo" == tv ]]; then
    alpha="${alpha:-1.0}"
    [[ "$alpha" =~ $number_pattern ]] || fail '--alpha must be a non-negative number, e.g. 1.0.'
else
    [[ -z "$alpha" ]] || fail '--alpha is only supported for tv.'
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
entrypoint="$repo_dir/baseline/code/muse_baselines/unlearn.py"
data_file="$repo_dir/baseline/data/enron/original_text/forget${suffix}.json"
tokenizer_dir="${tokenizer_dir:-$model_dir}"
out_dir="${out_dir:-$repo_dir/ckpt/enron/$algo/forget_$forget_rate}"
if [[ "$algo" == tv ]]; then
    # Keep the intermediate directory adjacent to the final output.
    while [[ "$out_dir" != / && "$out_dir" == */ ]]; do
        out_dir="${out_dir%/}"
    done
    [[ ! -e "${out_dir}_ft" || -d "${out_dir}_ft" ]] || fail "TV intermediate output path is not a directory: ${out_dir}_ft"
fi
if [[ "$needs_positive" == true ]]; then
    positive_data_file="${positive_data_file:-$repo_dir/baseline/data/enron/idk_text/forget${suffix}_idk.json}"
fi

[[ -d "$model_dir" ]] || fail "Model directory does not exist: $model_dir"
[[ -d "$tokenizer_dir" ]] || fail "Tokenizer directory does not exist: $tokenizer_dir"
[[ -r "$entrypoint" && -f "$entrypoint" ]] || fail "Training entry point is missing or unreadable: $entrypoint"
[[ -r "$data_file" && -f "$data_file" ]] || fail "Forget set is missing or unreadable: $data_file"
[[ ! -e "$out_dir" || -d "$out_dir" ]] || fail "Output path is not a directory: $out_dir"

check_data_file() {
    local path="$1"
    local label="$2"
    [[ -r "$path" && -f "$path" && -s "$path" ]] || fail "$label file is missing, empty, or unreadable: $path"
    case "$path" in
        *.txt|*.json) ;;
        *) fail "$label file must be .txt or .json: $path" ;;
    esac
}

if [[ "$needs_retain" == true ]]; then
    check_data_file "$retain_data_file" 'Retain data'
fi
if [[ "$needs_positive" == true ]]; then
    check_data_file "$positive_data_file" 'Positive data'
fi

python_bin="${PYTHON_BIN:-python}"
cmd=(
    "$python_bin" "$entrypoint"
    --algo "$algo"
    --model_dir "$model_dir"
    --tokenizer_dir "$tokenizer_dir"
    --data_file "$data_file"
    --out_dir "$out_dir"
    --max_len "$max_len"
    --epochs "$epochs"
    --lr "$lr"
    --per_device_batch_size "$per_device_batch_size"
)

if [[ "$needs_retain" == true ]]; then
    cmd+=(--retain_data_file "$retain_data_file")
fi
if [[ "$needs_positive" == true ]]; then
    cmd+=(--positive_data_file "$positive_data_file")
fi
if [[ "$algo" == tv ]]; then
    cmd+=(--alpha "$alpha")
fi

printf 'Unlearning command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$dry_run" == true ]]; then
    exit 0
fi

command -v "$python_bin" >/dev/null 2>&1 || fail "Interpreter not found: $python_bin. Set PYTHON_BIN to your training environment's interpreter."
exec "${cmd[@]}"
