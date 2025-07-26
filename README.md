# KnowUnDo

`KnowUnDo_privacy` and `processed_privacy` both are KnowUnDo data
folders.

## Data processing

You can generate the dataset for LLaMA-Factory by running:

``` bash
jq '.[0].train | map({question: .text, answer: .labels})' ./KnowUnDo_privacy/retention.json > ./processed_privacy/original/retention.json
jq '.[0].train | map({question: .text, answer: .labels})' ./KnowUnDo_privacy/unlearn.json > ./processed_privacy/original/unlearn.json
```

## Forget data sampling

You can use `sample.py` to randomly sample data for forget.

``` bash
python3 sample.py --help
usage: sample.py [-h] [--sample_size SAMPLE_SIZE] [--sample_ratio SAMPLE_RATIO] [--no_seed] [--seed SEED] input_file sampled_data_file unsampled_data_file

Sample data from a JSON file.

positional arguments:
  input_file            Path to the input JSON file.
  sampled_data_file     Path to save the sampled data.
  unsampled_data_file   Path to save the unsampled data.

options:
  -h, --help            show this help message and exit
  --sample_size, -ss SAMPLE_SIZE
                        Number of items to sample.
  --sample_ratio, -sr SAMPLE_RATIO
                        Fraction of items to sample.
  --no_seed             If set, do not freeze the random state for reproducibility.
  --seed SEED           Seed for random sampling. Default is 42.
```

For example:

``` bash
python3 sample.py processed_privacy/original/unlearn.json processed_privacy/sampled/unlearn_0.2.json processed_privacy/sampled/unknown_0.2.json
```

### Format data to text format

> The code works with JSON data that contains the `question` and
> `answer` fields.

``` bash
jq '[.[] | "Question: \(.question)\nAnswer: \(.answer)"]' ./processed_privacy/sampled/unlearn_0.2.json > ./processed_privacy/sampled_text/unlearn_0.2.json
```

## ICL data sampling

Sample 10 data from `retention.json`:

``` bash
python3 sample.py -ss 10 processed_privacy/original/retention.json processed_privacy/sampled/retention_qa_icl.json /dev/null
```

Sample 10 data from `unlearn.json`:

``` bash
python3 sample.py -ss 10 processed_privacy/original/unlearn.json processed_privacy/sampled/unlearn_qa_icl.json /dev/null
```

## Janus finetuning data sampling

Sample 50 data from `unlearn.json`:

``` bash
python3 sample.py -ss 50 processed_privacy/original/unlearn.json processed_privacy/sampled/janus50.json /dev/null
```

# IDK

The folder `idk/` is for IDK unlearning data.
