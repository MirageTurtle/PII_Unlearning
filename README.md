# PrivUn: Privacy Unlearning Robustness Benchmark

This repository contains the analysis code and feature probes for a PII unlearning robustness benchmark under three attack settings. The benchmark studies whether private information can still be recovered after unlearning, and analyzes how that residual leakage is reflected in model internals such as gradients, hidden representations, CKA similarity, and graph-based relations.

## What Is In This Repository

- `baseline/`
  Implementations and checkpoints for existing unlearning methods.
  Fine-tuning scripts
  Evaluation pipelines for the benchmark
  Typical baseline methods in our setup include methods such as `NPO`, `GA`, `RL`, `TV`, `RMU`, `DPO`, `IDK`, `WHP`, etc.
- `analysis/`
  Correlation analysis scripts and shared utilities.
- `get_memorization.py`
  Computes memorization-related / forgetting scores.
- `get_gradients.py`
  Extracts gradient-based features for unlearning models.
- `get_representation.py`
  Extracts hidden-layer activation / representation features.
- `get_cka.py`
  Compute CKA-based similarity analyses across models or datasets.
- `get_knowledge_graph.py`
  Builds sender-recipient graph statistics such as personalized PageRank.

## Repository Workflow

The intended workflow is:

1. Run an unlearning baseline from the `baseline/` folder.
2. Generate forgotten / recovered predictions on the benchmark splits.
3. Use this repository to extract internal features:
   `gradient`, `hidden-layer activation`, `CKA`, `knowledge-graph association`, and forgetting scores.
4. Run the correlation scripts in `analysis/` to measure how those features align with forgetting behavior.

## Running Unlearning Baselines

Run [baseline/unlearn.py](baseline/unlearn.py) to unlearn a model, passing the target model directory with `--model_dir` (the tokenizer is loaded from the same directory unless `--tokenizer_dir` is provided), using `--algo` to select a method (case-insensitive), supplying the method's training data with `--data_file` and the output model directory with `--out_dir`.

`--retain_data_file` is needed for GDR/KLR variants and RMU, and `--positive_data_file` for DPO methods. `--alpha` sets the non-negative scaling coefficient for TV method.

For RMU, there are some special arguments: `--rmu_layer_id` selects the zero-based decoder block used for the activation loss. `--rmu_layer_ids` specifies the blocks to update (space-separated). `--rmu_steering_coeff` controls the norm of RMU's random forget target, while `--rmu_retain_weight` controls the frozen-target retain loss.

An example to run GA:

```bash
python3 baseline/unlearn.py \
  --algo ga \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget02.json \
  --out_dir ./ckpt/enron/ga/forget_0.2
```

## Evaluating Baseline Checkpoints

[baseline/evaluate.py](baseline/evaluate.py) measures privacy recovery and utility. Provide `--forget_file`, `--retain_file`,
or both, plus the corresponding ICL files for few-shot evaluation.

```bash
python3 baseline/evaluate.py \
  --model_dir ./ckpt/enron/npo_gdr/forget_0.2 \
  --forget_file baseline/data/enron/original_qa/forget02.json \
  --retain_file ./data/retain_qa.json \
  --forget_icl_file ./data/forget_10_shot.json \
  --retain_icl_file ./data/retain_10_shot.json \
  --output_file ./results/npo_gdr_scores.json
```

Use `--k_shot 0` for zero-shot only. Add `--summary_file scores.csv` when a
single-row table summary is useful, or `--dry-run` to validate inputs without
loading the model.

## Analysis Scripts

The main correlation scripts are in `analysis/`: These scripts compute Pearson and Spearman correlation between forgetting scores and different hidden or structural features.

## Feature Probes

### Gradient Features

Use [get_gradients.py] to compute pair-wise gradient similarity against a reference set.

Example:

```bash
python get_gradients.py \
  --model_name /path/to/model \
  --recovered "data/unknown set/nonenron/0.2_recovered.json" \
  --forgotten "data/unknown set/nonenron/0.2_forgotten.json" \
  --forget_set "data/forget set/forget_0.2.json"
```

### Hidden-Layer Representations

Use [get_representation.py] choose the target hidden layer with `--layerid`.

Example:

```bash
python get_representation.py \
  --model_name /path/to/model \
  --recovered "data/unknown set/nonenron/0.2_recovered.json" \
  --forgotten "data/unknown set/nonenron/0.2_forgotten.json" \
  --forget_set "data/forget set/forget_0.2.json" \
  --layerid 28
```

### CKA

Use [get_cka.py] to compare layer-wise activation geometry across models, datasets, or training settings.

### Memorization / Forgetting Score

Use [get_memorization.py] to compute sequential token-level memorization-related scores used as forgetting indicators.

### Knowledge Graph Features

Use [get_knowledge_graph.py] to build the sender-recipient graph and compute association statistics such as personalized PageRank.

## Correlation Analysis

After generating the feature files in `results/`, run the analysis scripts.

Gradient correlation:

```bash
python -m analysis.gradient_correlation
```

Representation correlation:

```bash
python -m analysis.representation_correlation --layerid 28 --model npo
```

Knowledge graph correlation:

```bash
python -m analysis.knowledge_graph_correlation --model npo
```

## Notes

- Many scripts currently contain default local model paths. You will likely need to replace them with paths in your own environment.
- The analysis code assumes the feature extraction outputs follow the naming conventions already used in `results/`.
- The benchmark and the companion baseline folder are designed to be used together: the baseline folder produces unlearned models, and this repository analyzes their robustness.

## Citation

If you use this benchmark or build on this analysis code, please cite the corresponding project paper or repository once public metadata is available.
