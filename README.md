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

Train the target model as described in the paper and prepare your training
environment before running unlearning. Pass the local Hugging Face model directory with
`--model_dir`; the tokenizer is loaded from the same directory unless
`--tokenizer_dir` is provided.

Use `--algo` to select a method. Names are case-insensitive; the default is `ga`.
Run [baseline/unlearn.py](baseline/unlearn.py) directly, supplying the method's
training data with `--data_file` and the output model directory with `--out_dir`.
Algorithm implementations and shared training utilities live in
[baseline/core/](baseline/core/).

| Algorithm | Retain data | Positive data |
| --- | --- | --- |
| `ga` | Not used | Not used |
| `ga_gdr` | Required | Not used |
| `ga_klr` | Required | Not used |
| `npo` | Not used | Not used |
| `npo_gdr` | Required | Not used |
| `npo_klr` | Required | Not used |
| `dpo` | Not used | Required |
| `dpo_gdr` | Required | Required |
| `dpo_klr` | Required | Required |
| `tv` | Not used | Not used |
| `idk` | Not used | Not used |
| `rl` | Not used | Not used |
| `rm` | Not used | Not used |
| `whp` | Not used | Not used |

Supply `--retain_data_file` for GDR and KLR variants and `--positive_data_file`
for DPO methods. Positive data can use the matching bundled Enron IDK file or
your own positive responses. Forget, retain, and positive inputs accept `.txt`
or `.json` files; JSON must contain a list of strings or objects with a `text`
field. Positive data must match the forget set in sample count and question
order. Retain data is only accepted by GDR/KLR variants, and positive data is
only accepted by DPO methods.

`idk`, `rl`, `rm`, and `whp` use the shared [fine-tuning function](baseline/core/finetune.py)
and save the fine-tuned model directly to `--out_dir`. Supply prepared training
text through `--data_file`: question-refusal examples for IDK, questions with
randomly reassigned answers for RL, or replacement text for WHP. `rm` follows the
same training path as `rl` with its own prepared dataset. The WHP route
implements SFT on externally prepared text, using that text for both inputs and
labels. This is the simplified text-based variant; preparation of WHP data is a
separate step.

Examples from the repository root (provide your own retain data for the second
command):

```bash
# GA
python3 baseline/unlearn.py \
  --algo ga \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget02.json \
  --out_dir ./ckpt/enron/ga/forget_0.2

# NPO with a retain-data loss
python3 baseline/unlearn.py \
  --algo npo_gdr \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget02.json \
  --retain_data_file ./data/retain.txt \
  --out_dir ./ckpt/enron/npo_gdr/forget_0.2

# DPO with the bundled IDK positive responses
python3 baseline/unlearn.py \
  --algo dpo \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget05.json \
  --positive_data_file baseline/data/enron/idk_text/forget05_idk.json \
  --out_dir ./ckpt/enron/dpo/forget_0.5

# TV with a task-vector scaling coefficient
python3 baseline/unlearn.py \
  --algo tv \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget02.json \
  --out_dir ./ckpt/enron/tv/forget_0.2 \
  --alpha 1.0

# IDK supervised fine-tuning on prepared refusal examples
python3 baseline/unlearn.py \
  --algo idk \
  --model_dir ./models/target \
  --data_file baseline/data/enron/idk_text/forget02_idk.json \
  --out_dir ./ckpt/enron/idk/forget_0.2

# Check DPO settings and paths without loading a model or starting training
python3 baseline/unlearn.py \
  --algo dpo \
  --model_dir ./models/target \
  --data_file baseline/data/enron/original_text/forget05.json \
  --positive_data_file baseline/data/enron/idk_text/forget05_idk.json \
  --out_dir ./ckpt/enron/dpo/forget_0.5 \
  --dry-run
```

Each invocation trains one model unless `--dry-run` is supplied. Data and output
paths are always explicit; the examples above use the bundled Enron files.

TV first fine-tunes the target on the forget set, saving an intermediate model
to `<out_dir>_ft`. It then subtracts the scaled fine-tuning weight difference
from the target and saves the final model to `<out_dir>`. Use `--alpha` to set
the non-negative scaling coefficient (default: `1.0`); this option is used only
by TV. The shared training options below control TV's fine-tuning
step. The final TV directory currently saves model files; use the original
tokenizer directory when loading it.

Training defaults are 5 epochs, learning rate `1e-5`, per-device batch size 2,
and maximum sequence length 4096. Set `--epochs`, `--lr`,
`--per_device_batch_size`, and `--max_len` to match your experiment configuration.
NPO, DPO, and KLR methods automatically load a reference copy of the target model.
`--resume_from_checkpoint` is supported only by GA, NPO, and DPO methods and
their GDR/KLR variants.

Use `--help` for all options or append `--dry-run` to validate arguments and paths
and print the resolved settings as JSON. A dry run does not load models, create
output directories, or start training; it checks file availability, not model
compatibility or dataset contents. All relative paths are resolved from the
current working directory.

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
