#!/bin/bash
set -e

MODELS=(
  "./Llama-3.2-3B"
  "./target"
  "./target_npo"
  "/data2/chxiaoyi/models/enron_rmu05_layerid20"
  "/data2/chxiaoyi/models/enron_rmuv0907_02"
  "/data2/chxiaoyi/models/enron_rmuv0907_05"
  )

OUTDIR="results_mmlu"
mkdir -p ${OUTDIR}

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename ${MODEL})
  OUTFILE=${OUTDIR}/${NAME}.json

  if [ -f "${OUTFILE}" ]; then
    echo "[SKIP] ${NAME} already exists"
    continue
  fi

  echo "[RUN] Evaluating ${NAME}"

  lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},dtype=float16 \
    --tasks mmlu \
    --device cuda \
    --batch_size auto \
    --output_path ${OUTFILE}

  echo "[DONE] ${NAME}"
done
