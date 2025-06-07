# Data processing

You can generate the dataset for LLaMA-Factory by running:

``` bash
jq '.[0].train' ./KnowUnDo_privacy/retention.json > ./processed_privacy/retention.json
jq '.[0].train' ./KnowUnDo_privacy/unlearn.json > ./processed_privacy/unlearn.json
```
