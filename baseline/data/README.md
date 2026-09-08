# Enron

## IDK data

``` bash
python3 idk/generate_idk_data.py enron/original_qa/forget02.json idk/idontknow.jsonl enron/idk_qa/forget02_idk.json
python3 idk/generate_idk_data.py enron/original_qa/forget05.json idk/idontknow.jsonl enron/idk_qa/forget05_idk.json
mkdir -p enron/idk_text/
jq '[{"text": .[] | "Question: \(.question)\nAnswer: \(.answer)"}]' enron/idk_qa/forget02_idk.json > enron/idk_text/forget02_idk.json
jq '[{"text": .[] | "Question: \(.question)\nAnswer: \(.answer)"}]' enron/idk_qa/forget05_idk.json > enron/idk_text/forget05_idk.json
```

## sample 0.1 and 0.05 data

``` bash
python3 sample.py -sr 0.1 enron/original_qa/forget10.json enron/original_qa/forget01.json /dev/null
python3 sample.py -sr 0.05 enron/original_qa/forget10.json enron/original_qa/forget005.json /dev/null
```

## IDK source data

``` bash
idk
├── generate_idk_data.py  # the code for generating idk data
└── idontknow.jsonl       # the 100 idk source data
```
