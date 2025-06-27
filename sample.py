import random
from pathlib import Path
import json
import argparse


def read_json(file_path: Path) -> list:
    """Read a JSON file and return its content as a list."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sample_data(data: list, sample_size: int | float) -> tuple[list, list]:
    """Sample a specified number of items from the data."""
    if isinstance(sample_size, float):
        sample_size = int(len(data) * sample_size)
    # return random.sample(data, sample_size)
    # sample `sample_size` items from `data` without replacement
    # return sampled data and unsampled data
    sampled_data = random.sample(data, sample_size)
    unsampled_data = [item for item in data if item not in sampled_data]
    return sampled_data, unsampled_data


def freeze_random_state(seed: int = 42) -> tuple:
    """Freeze the random state for reproducibility."""
    random.seed(seed)
    return random.getstate()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sample data from a JSON file.")
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file.")
    parser.add_argument(
        "sampled_data_file", type=Path, help="Path to save the sampled data."
    )
    parser.add_argument(
        "unsampled_data_file", type=Path, help="Path to save the unsampled data."
    )
    parser.add_argument(
        "--sample_size",
        "-s",
        type=float,
        default=0.2,
        help="Fraction or number of items to sample.",
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
    """Main function to read, sample, and write data."""
    args = parse_args()

    # Read the input JSON file
    data = read_json(args.input_file)

    # Freeze the random state if no_seed is not set
    if not args.no_seed:
        freeze_random_state(args.seed)

    # Sample the data
    sampled_data, unsampled_data = sample_data(data, args.sample_size)

    # Write the sampled data to the output file
    with args.sampled_data_file.open("w", encoding="utf-8") as file:
        json.dump(sampled_data, file, ensure_ascii=False, indent=4)
    print(f"[INFO] Sampled {len(sampled_data)} items from {len(data)} total items.")

    # Write the unsampled data to a separate file
    with args.unsampled_data_file.open("w", encoding="utf-8") as file:
        json.dump(unsampled_data, file, ensure_ascii=False, indent=4)
    print(
        f"[INFO] Unsampled {len(unsampled_data)} items saved to {args.unsampled_data_file}."
    )


if __name__ == "__main__":
    main()
