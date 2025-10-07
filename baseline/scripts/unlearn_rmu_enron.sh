#!/usr/bin/bash

if ! command -v conda &>/dev/null; then
    echo 'command `conda` is not found, env error!'
    exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda activate rmu; then
    echo "Failed to activate rmu conda environment"
    exit 1
fi

cd /path/to/wmdp/ || {
    echo "Failed to cd to wmdp directory"
    exit 1
}

echo "---- $(date '+%Y-%m-%d %H:%M:%S') ----"
MODEL_NAME_OR_PATH="/path/to/target"
REFERENCE_NAME_OR_PATH="/path/to/Llama-3.2-3B"
USE_CUSTOM_RMU=true
EPOCH_NUM=5
BATCH_SIZE=2
LAYER_ID=27
LAYER_IDS=20,21,22,23,24,25,26,27
PARAM_IDS=0,1,2,3,4,5,6,7,8
MAX_NUM_BATCHES=400
RETAIN_DATA_PATH="/path/to/muse_bench/data/news/raw/retain1.txt"
FORGET_DATA_PATH="/path/to/PII_Unlearning/enron/original_text/forget02.json"
STEERING_COEFFS="300"
ALPHA="1600"
MIN_LEN=50
LR="5e-5"
SEED=42
OUTPUT_DIR="models/enron_rmu_opt_02"
if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
    echo "Model directory $MODEL_NAME_OR_PATH does not exist."
    exit 1
fi

CMD=(
    python3 -m rmu.unlearn
    --model_name "$MODEL_NAME_OR_PATH"
    --batch_size "$BATCH_SIZE"
    --layer_id "$LAYER_ID"
    --layer_ids "$LAYER_IDS"
    --param_ids "$PARAM_IDS"
    --max_num_batches "$MAX_NUM_BATCHES"
    --retain_data_path "$RETAIN_DATA_PATH"
    --forget_data_path "$FORGET_DATA_PATH"
    --read_data_from_files
    --steering_coeffs "$STEERING_COEFFS"
    --alpha "$ALPHA"
    --min_len "$MIN_LEN"
    --lr "$LR"
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
)
if [ "$USE_CUSTOM_RMU" = true ]; then
    CMD+=(
        --reference_model_name_or_path "$REFERENCE_NAME_OR_PATH"
        --custom_rmu
        --epochs "$EPOCH_NUM"
    )
fi

# echo cmd first
echo "Running command: ${CMD[*]}"

"${CMD[@]}"

echo "---- $(date '+%Y-%m-%d %H:%M:%S') ----"
