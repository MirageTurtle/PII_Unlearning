# Data processing

You can generate the dataset for LLaMA-Factory by running:

``` bash
jq '.[0].train' ./KnowUnDo_privacy/retention.json > ./processed_privacy/retention.json
jq '.[0].train' ./KnowUnDo_privacy/unlearn.json > ./processed_privacy/unlearn.json
```

# Forget data sampling

You can use `sample.py` to randomly sample data for forget.

``` bash
python3 sample.py --help
usage: sample.py [-h] [--sample_size SAMPLE_SIZE] [--no_seed] [--seed SEED] input_file output_file

Sample data from a JSON file.

positional arguments:
  input_file            Path to the input JSON file.
  output_file           Path to save the sampled data.

options:
  -h, --help            show this help message and exit
  --sample_size, -s SAMPLE_SIZE
                        Fraction or number of items to sample.
  --no_seed             If set, do not freeze the random state for reproducibility.
  --seed SEED           Seed for random sampling. Default is 42.
```

For example:

``` bash
python3 sample.py processed_privacy/unlearn.json processed_privacy/unlearn_0.2.json
```
