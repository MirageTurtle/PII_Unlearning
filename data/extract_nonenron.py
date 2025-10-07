#!/usr/bin/env python3
"""
This script extracts non-enron data from a given enron data file.
"""

import argparse
import json
import os
from pathlib import Path
import re


def argparse_args():
    parser = argparse.ArgumentParser(
        description="Extract non-enron data from given enron data file."
    )
    parser.add_argument("enron_file", type=Path, help="Path to the enron data file.")
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to save the extracted non-enron data. Use '-' for stdout.",
    )
    return parser.parse_args()


def extract_name_email(data: dict | str) -> dict[str, str]:
    """
    Extracts the name and email from the given data.
    Args:
            data (dict | str): The data containing name and email.
    Returns:
            dict[str, str]: A dictionary containing the name and email.
    """
    if isinstance(data, dict):
        try:
            if "name" in data and "email" in data:
                return {"name": data["name"], "email": data["email"]}
        except KeyError:
            raise ValueError('Key "name" or "email" not found in the provided data.')
    elif isinstance(data, str):
        # return data
        try:
            pattern = r"The email address of (.+) is ([\w\.-]+@[\w\.-]+)"
            match = re.search(pattern, data)
            if match:
                return {"name": match.group(1), "email": match.group(2)}
            else:
                raise ValueError("No name and email found in the provided string.")
        except re.error as e:
            raise ValueError(f"Regex error: {e}")
    raise TypeError("Data must be a dictionary or a string.")


def check_nonenron(data: dict) -> bool:
    return not data["email"].endswith("@enron.com")


def extract_nonenron_data(enron_data: list[str | dict]) -> list[dict]:
    """
    Extract non-enron data from the given enron data dictionary.

    Args:
        enron_data (list[str | dict]): The enron data from which to extract non-enron data.

    Returns:
        list[dict]: A list of dictionaries containing the extracted non-enron data.
    """
    non_enron_data = []
    for item in enron_data:
        try:
            name_email = extract_name_email(item)
            if name_email and check_nonenron(name_email):
                non_enron_data.append(name_email)
        except (ValueError, TypeError) as e:
            print(f"Error processing item {item}: {e}", file=os.sys.stderr)
            continue
    return non_enron_data


def main():
    args = argparse_args()
    enron_file = args.enron_file
    output_file = args.output_file

    if not enron_file.is_file():
        raise FileNotFoundError(f"Enron file {enron_file} does not exist.")

    if enron_file.suffix == ".json":
        with open(enron_file, "r", encoding="utf-8") as f:
            enron_data = json.load(f)
    elif enron_file.suffix == ".txt":
        with open(enron_file, "r", encoding="utf-8") as f:
            enron_data = f.readlines()
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json or .txt file."
        )

    non_enron_data = extract_nonenron_data(enron_data)

    if output_file == Path("-"):
        print(json.dumps(non_enron_data, indent=2))
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(non_enron_data, f, indent=2)


if __name__ == "__main__":
    main()
