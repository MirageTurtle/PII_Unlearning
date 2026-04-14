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

