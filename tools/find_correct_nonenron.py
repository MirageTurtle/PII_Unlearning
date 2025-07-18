#!/usr/bin/env python3
"""
This file is for finding correct non-enron data from a bunch of qa log files.
"""

import argparse
import json
from pathlib import Path
import re


def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find correct non-enron data from a bunch of qa log files."
    )
    parser.add_argument(
        "qa_log_list_file",
        type=Path,
        help="Path to the file containing a list of QA log files.",
    )
    parser.add_argument(
        "correct_output_file",
        type=Path,
        help="Path to save the correct non-enron data. Use '-' for stdout.",
    )
    parser.add_argument(
        "incorrect_output_file",
        type=Path,
        help="Path to save the incorrect non-enron data. Use '-' for stdout.",
    )
    parser.add_argument(
        "--no_janus",
        action="store_true",
        help="If set, filter out data used in Janus.",
    )
    parser.add_argument(
        "--janus_file",
        type=Path,
        help="Path to the Janus data file. Required if --no_janus is set.",
    )
    return parser.parse_args()


def find_correct_nonenron_data_from_one_qa_log(
    qa_log_file: Path, no_janus: bool = False, janus_email_data: list[str] | None = None
) -> tuple[list[dict], list[dict]]:
    """
    Find correct non-enron data from a single QA log file.

    Args:
        qa_log_file (Path): Path to the QA log file.
        no_janus (bool): If True, filter out data used in Janus.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing two lists:
            - correct_nonenron_data: List of dictionaries containing correct non-enron data.
            - incorrect_nonenron_data: List of dictionaries containing incorrect non-enron data.
    """
    if qa_log_file.suffix != ".json":
        raise ValueError(f"Expected a JSON file, got {qa_log_file.suffix}")
    if no_janus and janus_email_data is None:
        raise ValueError("Janus email data must be provided if --no_janus is set.")
    correct_nonenron_data = []
    incorrect_nonenron_data = []
    qa_log_data = json.loads(qa_log_file.read_text())
    for item in qa_log_data:
        gt = item["gt"]
        # if non-enron/correct/not used in janus
        if (
            gt.endswith("@enron.com")
            or gt.lower() not in item["response"].lower()
            or (no_janus and gt in janus_email_data)
        ):
            incorrect_nonenron_data.append(item)
        else:
            correct_nonenron_data.append(item)
    return correct_nonenron_data, incorrect_nonenron_data


def find_correct_nonenron_data(
    qa_log_list_file: Path,
    no_janus: bool = False,
    janus_email_data: list[str] | None = None,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """
    Find correct non-enron data from a list of QA log files.

    Args:
        qa_log_list_file (Path): Path to the file containing a list of QA log files.
        no_janus (bool): If True, filter out data used in Janus.

    Returns:
        tuple[dict[str, list[dict]], dict[str, list[dict]]]: A tuple containing two dictionaries:
            - all_correct_nonenron_data: Dictionary with QA log file names as keys and lists of correct non-enron data as values.
            - all_incorrect_nonenron_data: Dictionary with QA log file names as keys and lists of incorrect non-enron data as values.
    """
    if not qa_log_list_file.exists():
        raise FileNotFoundError(f"File {qa_log_list_file} does not exist.")

    with qa_log_list_file.open("r") as f:
        qa_log_files = [Path(line.strip()) for line in f if line.strip()]

    all_correct_nonenron_data = {}
    all_incorrect_nonenron_data = {}
    for qa_log_file in qa_log_files:
        correct_nonenron_data, incorrect_nonenron_data = (
            find_correct_nonenron_data_from_one_qa_log(
                qa_log_file, no_janus, janus_email_data
            )
        )
        all_correct_nonenron_data[str(qa_log_file)] = correct_nonenron_data
        all_incorrect_nonenron_data[str(qa_log_file)] = incorrect_nonenron_data
    return all_correct_nonenron_data, all_incorrect_nonenron_data


def main():
    args = argparse_args()
    qa_log_list_file = args.qa_log_list_file
    correct_output_file = args.correct_output_file
    incorrect_output_file = args.incorrect_output_file
    no_janus = args.no_janus
    janus_email_data = None

    if no_janus:
        if not args.janus_file or not args.janus_file.is_file():
            raise FileNotFoundError(f"Janus file {args.janus_file} does not exist.")
        janus_email_data = json.loads(args.janus_file.read_text())
        janus_email_data = [item["answer"] for item in janus_email_data]

    all_correct_nonenron_data, all_incorrect_nonenron_data = find_correct_nonenron_data(
        qa_log_list_file, no_janus=True, janus_email_data=janus_email_data
    )

    for output_file, data in zip(
        [correct_output_file, incorrect_output_file],
        [all_correct_nonenron_data, all_incorrect_nonenron_data],
    ):
        if output_file == Path("-"):
            print(json.dumps(data, indent=2))
        else:
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
