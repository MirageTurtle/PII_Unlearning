"""Evaluate forgetting and retained QA performance with one model load."""

import argparse
import csv
import json
from pathlib import Path


def load_qa_file(path):
    """Load QA examples, rejecting empty answers that trivially match any output."""
    with Path(path).open(encoding="utf-8") as stream:
        examples = json.load(stream)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Expected a nonempty JSON list: {path}")
    if any(
        not isinstance(item, dict)
        or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("question", "answer")
        )
        for item in examples
    ):
        raise ValueError(
            f"Each example needs nonempty question and answer strings: {path}"
        )
    return examples


def build_prompt(item, demos):
    prefix = "".join(
        f"Question: {demo['question']}\nAnswer: {demo['answer']}\n\n" for demo in demos
    )
    return f"{prefix}Question: {item['question']}\nAnswer: "


def generate_completions(model, tokenizer, prompts, batch_size, max_new_tokens, desc):
    import torch
    from tqdm.auto import tqdm

    completions = []
    input_device = model.get_input_embeddings().weight.device
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        encoded = tokenizer(
            prompts[start : start + batch_size], return_tensors="pt", padding=True
        ).to(input_device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Slice after the padded input width so neither prompts nor demos are scored.
        completion_ids = generated[:, encoded["input_ids"].shape[1] :]
        completions.extend(
            tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        )
    return completions


def score_completions(
    examples, prompts, completions, task, case_sensitive=False, scorer=None
):
    """Preserve the reference scripts' full-completion match and first-line ROUGE."""
    if not examples or not len(examples) == len(prompts) == len(completions):
        raise ValueError(
            "Examples, prompts, and completions must have equal nonzero lengths."
        )
    if task not in ("forget", "retain"):
        raise ValueError(f"Unknown evaluation task: {task}")
    if task == "retain" and scorer is None:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    for item, prompt, completion in zip(examples, prompts, completions):
        answer = item["answer"]
        result = {
            "question": item["question"],
            "prompt": prompt,
            "gt": answer,
            "output": completion,
        }
        if task == "forget":
            result["correct"] = (
                answer in completion
                if case_sensitive
                else answer.casefold() in completion.casefold()
            )
        else:
            predicted = completion.strip().split("\n")[0].strip()
            result.update(
                predicted=predicted,
                rougeL=scorer.score(answer, predicted)["rougeL"].fmeasure,
            )
        results.append(result)
    output = {"total": len(results), "results": results}
    if task == "forget":
        correct = sum(item["correct"] for item in results)
        accuracy = correct / len(results)
        output.update(correct=correct, accuracy=accuracy, forgetting_rate=1 - accuracy)
    else:
        output["average"] = sum(item["rougeL"] for item in results) / len(results)
    return output


def evaluate_task(model, tokenizer, examples, demos, task, args):
    output = {"zero_shot": None, "few_shot": None, "demos": demos}
    for condition in ("zero_shot", "few_shot"):
        if condition == "few_shot" and args.k_shot == 0:
            continue
        selected_demos = demos if condition == "few_shot" else []
        # The retain reference excludes repeated questions; the forget reference
        # deliberately uses one fixed demonstration prefix for every question.
        per_example_demos = [
            [demo for demo in selected_demos if demo["question"] != item["question"]]
            if task == "retain"
            else selected_demos
            for item in examples
        ]
        prompts = [
            build_prompt(item, prefix)
            for item, prefix in zip(examples, per_example_demos)
        ]
        completions = generate_completions(
            model,
            tokenizer,
            prompts,
            args.batch_size,
            args.max_new_tokens,
            f"{task}: {condition}",
        )
        scored = score_completions(
            examples, prompts, completions, task, args.case_sensitive
        )
        for result, prefix in zip(scored["results"], per_example_demos):
            result["demo_count"] = len(prefix)
        output[condition] = scored
    return output


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "The tokenizer needs a padding token or EOS token for batched generation."
        )
    tokenizer.padding_side = "left"
    kwargs = {"torch_dtype": "auto"}
    if args.device_map_auto:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    if model.config.is_encoder_decoder:
        raise ValueError("This evaluator expects a decoder-only causal language model.")
    if not args.device_map_auto:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, tokenizer


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir", required=True, help="Local Hugging Face model directory."
    )
    parser.add_argument(
        "--tokenizer_dir",
        help="Defaults to --model_dir; set explicitly for checkpoints without a tokenizer.",
    )
    parser.add_argument(
        "--forget_file", help="QA JSON to score by answer-substring match."
    )
    parser.add_argument("--retain_file", help="QA JSON to score by ROUGE-L F1.")
    parser.add_argument(
        "--forget_icl_file",
        help="QA demonstrations for forget-set few-shot evaluation.",
    )
    parser.add_argument(
        "--retain_icl_file",
        help="QA demonstrations for retain-set few-shot evaluation.",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=10,
        help="Use the first k demonstrations; 0 runs zero-shot only (default: 10).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Case-sensitive forget answer matching (default: case-insensitive).",
    )
    parser.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Use device_map='auto' (requires accelerate); default: one CUDA device or CPU.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="JSON containing settings, summary scores, and per-example results.",
    )
    parser.add_argument(
        "--summary_file",
        help="Optional one-row CSV containing the same summary scores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and QA data without loading the model or writing outputs.",
    )
    args = parser.parse_args(argv)
    if not args.forget_file and not args.retain_file:
        parser.error("Supply --forget_file, --retain_file, or both.")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.k_shot < 0:
        parser.error(
            "Batch size and max new tokens must be positive; k_shot must be non-negative."
        )
    for task in ("forget", "retain"):
        data = getattr(args, f"{task}_file")
        demos = getattr(args, f"{task}_icl_file")
        if data and args.k_shot > 0 and not demos:
            parser.error(f"Provide --{task}_icl_file or use --k_shot 0.")
        if demos and (not data or args.k_shot == 0):
            parser.error(f"--{task}_icl_file requires --{task}_file and --k_shot > 0.")
    args.tokenizer_dir = args.tokenizer_dir or args.model_dir
    for name, value in vars(args).items():
        if not name.endswith(("_file", "_dir")) or value is None:
            continue
        if not value:
            parser.error(f"--{name} must not be empty.")
        path = Path(value).expanduser().resolve()
        setattr(args, name, str(path))
        if name.endswith("_dir"):
            if not path.is_dir():
                parser.error(f"Directory does not exist: {path}")
        elif name in ("output_file", "summary_file"):
            if path.exists() and not path.is_file():
                parser.error(f"Output path is not a file: {path}")
        elif not path.is_file():
            parser.error(f"QA file does not exist: {path}")
    inputs = {
        args.forget_file,
        args.retain_file,
        args.forget_icl_file,
        args.retain_icl_file,
    }
    if (
        args.output_file in inputs
        or args.summary_file is not None
        and args.summary_file in inputs
    ):
        parser.error("Output files must differ from the input datasets.")
    if args.output_file == args.summary_file:
        parser.error("JSON and CSV output paths must differ.")
    return args


def prepare_data(args):
    datasets = {}
    for task in ("forget", "retain"):
        path = getattr(args, f"{task}_file")
        if path is None:
            continue
        examples = load_qa_file(path)
        demos = []
        if args.k_shot:
            demos = load_qa_file(getattr(args, f"{task}_icl_file"))
            if len(demos) < args.k_shot:
                raise ValueError(
                    f"{task} ICL file contains {len(demos)} examples; need {args.k_shot}."
                )
            demos = demos[: args.k_shot]
        datasets[task] = (examples, demos)
    return datasets


def summarize(output):
    """Return raw [0, 1] scores; skipped evaluations stay null, never zero."""
    summary = {
        "model": output["config"]["model_dir"],
        "k_shot": output["config"]["k_shot"],
    }
    for condition in ("zero_shot", "few_shot"):
        forget = output["forget"][condition] if output["forget"] else None
        retain = output["retain"][condition] if output["retain"] else None
        summary[f"forget_accuracy_{condition}"] = forget["accuracy"] if forget else None
        summary[f"forgetting_rate_{condition}"] = (
            forget["forgetting_rate"] if forget else None
        )
        summary[f"retain_rougeL_{condition}"] = retain["average"] if retain else None
    return summary


def main(argv=None):
    args = get_args(argv)
    datasets = prepare_data(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": vars(args),
                    "counts": {task: len(data[0]) for task, data in datasets.items()},
                },
                indent=2,
            )
        )
        return
    # Fail on a missing metric dependency before loading a large checkpoint.
    if "retain" in datasets:
        from rouge_score import rouge_scorer  # noqa: F401

    model, tokenizer = load_model_and_tokenizer(args)
    output = {"config": vars(args), "forget": None, "retain": None}
    for task, (examples, demos) in datasets.items():
        output[task] = evaluate_task(model, tokenizer, examples, demos, task, args)
    output["summary"] = summarize(output)
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.summary_file:
        path = Path(args.summary_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output["summary"]))
            writer.writeheader()
            writer.writerow(output["summary"])
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved evaluation results to {args.output_file}")


if __name__ == "__main__":
    main()
