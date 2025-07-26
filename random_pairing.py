import random
from pathlib import Path
import json
import argparse


def read_json(file_path: Path) -> list:
    """Read a JSON file and return its content as a list."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_random_pairing_data(
    original_data: list[dict[str, str]], answer_key: str = "answer"
) -> list[dict[str, str]]:
    """
    Generate random pairing data by shuffling the answers.
    """
    answers = [item[answer_key] for item in original_data]
    random.shuffle(answers)

    paired_data = []
    for i, item in enumerate(original_data):
        new_item = item.copy()
        new_item[answer_key] = answers[i]
        paired_data.append(new_item)

    return paired_data


def freeze_random_state(seed: int = 42) -> tuple:
    """Freeze the random state for reproducibility."""
    random.seed(seed)
    return random.getstate()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random pairing data from a JSON file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file.")
    parser.add_argument(
        "output_file", type=Path, help="Path to save the generated pairing data."
    )
    parser.add_argument(
        "--no_seed",
        action="store_true",
        help="If set, do not freeze the random state for reproducibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read original data
    original_data = read_json(args.input_file)

    # Freeze the random state if no_seed is not set
    if not args.no_seed:
        freeze_random_state(args.seed)

    # Generate random pairing data
    paired_data = generate_random_pairing_data(original_data)

    # Save the generated pairing data to the output file
    with args.output_file.open("w", encoding="utf-8") as file:
        json.dump(paired_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
