import random
from pathlib import Path
import json
import argparse


def read_json(file_path: Path) -> list:
    """Read a JSON file and return its content as a list."""
    with file_path.open("r", encoding="utf-8") as file:
        # return json.load(file)
        first_line = file.readline().strip()
        if first_line.startswith(("[", "{")):
            if file_path.suffix == ".json":
                file.seek(0)
                return json.load(file)
            elif file_path.suffix == ".jsonl":
                return [json.loads(line) for line in file if line.strip()]
            else:
                raise ValueError("Unsupported file format.")
        else:
            file.seek(0)
            return [line.strip() for line in file if line.strip()]


def sample_idk(
    idk_data: list[str], sample_size: int, duplicates: bool = False
) -> list[str]:
    """Sample a specified number of items from the IDK data."""
    if duplicates:
        return random.choices(idk_data, k=sample_size)
    else:
        return random.sample(idk_data, sample_size)


def freeze_random_state(seed: int = 42) -> tuple:
    """Freeze the random state for reproducibility."""
    random.seed(seed)
    return random.getstate()


def generate_idk_data(
    original_data: list[dict[str, str]], idk_data: list[str], answer_key: str = "answer"
) -> list[dict[str, str]]:
    """
    Generate IDK data by sampling from the original data.
    Replace the answer with sampled IDK data.
    """
    generated_idk_data = []
    len_data = len(original_data)
    sampled_idk = sample_idk(idk_data, len_data, duplicates=True)
    for i, item in enumerate(original_data):
        new_item = item.copy()
        new_item[answer_key] = sampled_idk[i]
        generated_idk_data.append(new_item)
    return generated_idk_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate IDK data from original data."
    )
    parser.add_argument(
        "original_data_file", type=Path, help="Path to the original data JSON file."
    )
    parser.add_argument(
        "idk_data_file", type=Path, help="Path to the IDK data JSON file."
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the generated IDK data."
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="answer",
        help="Key for the answer in the original data.",
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
        help="Seed for random sampling. Default is 42.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Read original data and IDK data
    original_data = read_json(args.original_data_file)
    idk_data = read_json(args.idk_data_file)

    # Freeze the random state if no_seed is not set
    if not args.no_seed:
        freeze_random_state(args.seed)

    # Generate IDK data
    generated_idk_data = generate_idk_data(original_data, idk_data, args.answer_key)

    # Save the generated IDK data to the output file
    with args.output_file.open("w", encoding="utf-8") as file:
        json.dump(generated_idk_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
