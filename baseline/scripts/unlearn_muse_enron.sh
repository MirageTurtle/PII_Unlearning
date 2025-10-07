#!/usr/bin/env bash

CORPUS='enron'

FORGET="/path/to/PII_Unlearning/enron/original_text/forgetSUFFIX.json"
RETAIN="../data/news/raw/retain1.txt"
POSITIVE="/path/to/PII_Unlearning/enron/idk_text/forgetSUFFIX_idk.json"

TARGET_DIR='/path/to/target'

MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1
FT_EPOCHS=10
FT_LR='1e-5'

declare -A rate_to_suffix=(
    # ["rate"]="suffix"
    ["0.05"]="005"
    ["0.1"]="01"
    ["0.2"]="02"
    ["0.5"]="05"
    ["1.0"]="10"
)

# for algo in 'dpo' 'dpo_gdr' 'dpo_klr'; do
# for algo in 'ga' 'ga_gdr' 'ga_klr' 'npo' 'npo_gdr' 'npo_klr'; do
for algo in 'ga'; do
    # for forget_rate in "${!rate_to_suffix[@]}"; do
    # for forget_rate in "0.05" "0.1" "1.0"; do
    for forget_rate in "0.2" "0.5"; do
        suffix=${rate_to_suffix[$forget_rate]}
        # check if the forget_rate is in the array
        if [[ -z "$suffix" ]]; then
            echo "Forget rate $forget_rate not found in the mapping."
            echo "Continuing to the next rate."
            continue
        fi
        tmp_forget=$(sed "s/SUFFIX/$suffix/g" <<<$FORGET)
        tmp_positive=$(sed "s/SUFFIX/$suffix/g" <<<$POSITIVE)
        tmp_rate=$forget_rate
        echo "Unlearning with algo: $algo, forget rate: $forget_rate, forget file: $tmp_forget, out_dir: ./ckpt/$CORPUS/$algo/forget_$tmp_rate, positive file: $tmp_positive"

        cmd="python unlearn.py \
            --algo $algo \
            --model_dir $TARGET_DIR \
            --data_file $tmp_forget --retain_data_file $RETAIN \
            --out_dir "./ckpt/$CORPUS/$algo/forget_$tmp_rate" \
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE"
        if [[ $algo == *"dpo"* ]]; then
            cmd="$cmd --positive_data_file $tmp_positive"
        fi
        echo $cmd
        eval $cmd
    done
done
