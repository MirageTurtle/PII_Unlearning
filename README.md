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

## GA, NPO, DPO, and TV Baselines

Train the target model as described in the paper and prepare your training
environment before running unlearning. Pass the local Hugging Face model directory with
`--model_dir`; the tokenizer is loaded from the same directory unless
`--tokenizer_dir` is provided.

Use `--algo` to select a method. Names are case-insensitive; the default is `ga`.
All methods use the bundled Enron forget set selected by `--forget_rate`.

| Algorithm | Retain data | Positive data |
| --- | --- | --- |
| `ga` | Not used | Not used |
| `ga_gdr` | Required | Not used |
| `ga_klr` | Required | Not used |
| `npo` | Not used | Not used |
| `npo_gdr` | Required | Not used |
| `npo_klr` | Required | Not used |
| `dpo` | Not used | Bundled IDK data by default |
| `dpo_gdr` | Required | Bundled IDK data by default |
| `dpo_klr` | Required | Bundled IDK data by default |
| `tv` | Not used | Not used |

Supply `--retain_data_file` for GDR and KLR variants. DPO methods automatically
select the IDK text file for the chosen forget rate; use `--positive_data_file`
to supply your own positive responses. Retain and positive inputs accept `.txt`
or `.json` files; JSON must contain a list of strings or objects with a `text`
field. Custom positive data must match the forget set in sample count and
question order. Retain data is only accepted by GDR/KLR variants, and positive
data is only accepted by DPO methods.

Examples from the repository root (provide your own retain data for the second
command):

```bash
# GA
bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
  --algo ga \
  --model_dir ./models/target \
  --forget_rate 0.2

# NPO with a retain-data loss
bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
  --algo npo_gdr \
  --model_dir ./models/target \
  --forget_rate 0.2 \
  --retain_data_file ./data/retain.txt

# DPO with the bundled IDK positive responses
bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
  --algo dpo \
  --model_dir ./models/target \
  --forget_rate 0.5

# TV with a task-vector scaling coefficient
bash baseline/scripts/unlearn_ga_npo_dpo_tv.sh \
  --algo tv \
  --model_dir ./models/target \
  --forget_rate 0.2 \
  --alpha 1.0
```

The script supports Enron forget rates `0.2` and `0.5` (default: `0.2`).
Each invocation trains one model. The default output is
`ckpt/enron/ALGO/forget_RATE` under the repository root, using the lowercase
algorithm name; override it with `--out_dir`.

TV first fine-tunes the target on the forget set, saving an intermediate model
to `<out_dir>_ft`. It then subtracts the scaled fine-tuning weight difference
from the target and saves the final model to `<out_dir>`. Use `--alpha` to set
the non-negative scaling coefficient (default: `1.0`); this option is only
accepted for TV. The shared training options below control TV's fine-tuning
step. The final TV directory currently saves model files; use the original
tokenizer directory when loading it.

Launcher defaults follow the existing script: 10 epochs, learning rate `1e-5`,
per-device batch size 1, and maximum sequence length 2048. Set `--epochs`, `--lr`,
`--per_device_batch_size`, and `--max_len` to match your experiment configuration.
NPO, DPO, and KLR methods automatically load a reference copy of the target model.

Use `--help` for all options or append `--dry-run` to check paths and preview the
command without starting training. Set `PYTHON_BIN` to choose the interpreter in
your training environment. Data paths are located relative to the script;
user-supplied relative paths are resolved from the current working directory.

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
