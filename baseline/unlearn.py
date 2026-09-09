"""Command-line entry point for the baseline package."""

import argparse
import json
import math
import os
import sys
from os.path import basename, dirname
from os.path import join as pathjoin
from pathlib import Path


def main():
    args = get_args()

    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return

    if not __package__:
        # Direct script execution needs the repository root on the import path.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from baseline.core import finetune, it_unlearn, tv_unlearn

    if args.algo == "tv":
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir,
            args.data_file,
            ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
        )
        tv_unlearn(
            args.model_dir,
            args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha,
        )

    else:
        it_unlearn(
            args.model_dir,
            args.data_file,
            args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            positive_data_file=args.positive_data_file,
        )

    return


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument(
        "--algo",
        type=str.lower,
        choices=(
            "ga",
            "ga_gdr",
            "ga_klr",
            "npo",
            "npo_gdr",
            "npo_klr",
            "dpo",
            "dpo_gdr",
            "dpo_klr",
            "tv",
        ),
        default="ga",
        help="Unlearning method (case-insensitive; default: ga).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the target model's hf directory.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the forget set (.txt or .json).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to the output model directory.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=4096,
        help="max length of input ids fed to the model",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print resolved settings without loading models or training.",
    )

    # Gradient ascent & Gradient difference
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument(
        "--retain_data_file",
        type=str,
        default=None,
        help="Path to the retain set file. Required for *_gdr and *_klr.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Training learning rate (default: 1e-5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )

    # Task vector
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling coefficient scales the task vector if algo is task vector (tv).",
    )

    # DPO
    parser.add_argument(
        "--positive_data_file",
        type=str,
        default=None,
        help="Path to the positive data file. Required for DPO methods.",
    )

    args = parser.parse_args(argv)

    needs_retain = args.algo.endswith(("_gdr", "_klr"))
    needs_positive = args.algo.startswith("dpo")
    if needs_retain and not args.retain_data_file:
        parser.error(f"--retain_data_file is required for {args.algo}.")
    if not needs_retain and args.retain_data_file is not None:
        parser.error("--retain_data_file is only used by *_gdr and *_klr.")
    if not needs_positive and args.positive_data_file is not None:
        parser.error("--positive_data_file is only used by DPO methods.")

    if args.resume_from_checkpoint and args.algo == "tv":
        parser.error("Cannot resume from checkpoint for TV.")

    if needs_positive and not args.positive_data_file:
        parser.error("--positive_data_file is required for DPO methods.")

    if args.epochs < 1 or args.per_device_batch_size < 1 or args.max_len < 2:
        parser.error(
            "--epochs and --per_device_batch_size must be positive; --max_len must be at least 2."
        )
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be a finite positive number.")
    if args.algo == "tv" and (not math.isfinite(args.alpha) or args.alpha < 0):
        parser.error("--alpha must be a finite non-negative number.")

    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.model_dir

    for name in (
        "model_dir",
        "tokenizer_dir",
        "data_file",
        "retain_data_file",
        "positive_data_file",
        "out_dir",
    ):
        value = getattr(args, name)
        if value is None:
            continue
        if not value:
            parser.error(f"--{name} must not be empty.")
        path = Path(value).expanduser().resolve()
        setattr(args, name, str(path))
        if name in ("model_dir", "tokenizer_dir"):
            if not path.is_dir():
                parser.error(f"--{name} directory does not exist: {path}")
        elif name == "out_dir":
            if path.exists() and not path.is_dir():
                parser.error(f"--out_dir is not a directory: {path}")
        else:
            if path.suffix not in (".txt", ".json"):
                parser.error(f"--{name} must be a .txt or .json file: {path}")
            if (
                not path.is_file()
                or not os.access(path, os.R_OK)
                or path.stat().st_size == 0
            ):
                parser.error(f"--{name} file is missing, empty, or unreadable: {path}")

    if args.algo == "tv":
        ft_model_dir = Path(args.out_dir + "_ft")
        if ft_model_dir.exists() and not ft_model_dir.is_dir():
            parser.error(
                f"TV intermediate output path is not a directory: {ft_model_dir}"
            )

    return args


if __name__ == "__main__":
    main()
