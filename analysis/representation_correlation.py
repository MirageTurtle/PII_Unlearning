import argparse

try:
    from .correlation import (
        analyze_correlation,
        extract_series,
        match_mem_scores_by_text,
    )
except ImportError:
    from correlation import (
        analyze_correlation,
        extract_series,
        match_mem_scores_by_text,
    )


def main():
    parser = argparse.ArgumentParser(description="Representation Analysis")
    parser.add_argument(
        "--layerid",
        "-l",
        type=int,
        required=True,
        help="the layer ID of hidden layer representation",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="npo",
        help="the model to calculate the forgetting score",
    )
    args = parser.parse_args()

    mem_score_path = f"results/mem_score/unknown_forgotten_log_{args.model}.json"
    representation_path = f"results/representation/forgotten_unknown_unlearn_{args.layerid}.json"

    matches = match_mem_scores_by_text(
        mem_score_path,
        representation_path,
        metric_fields=["cosine_similarity"],
    )

    cosine_values, unlearn_values = extract_series(matches, "cosine_similarity")
    assert len(cosine_values) == len(unlearn_values)

    analyze_correlation(
        cosine_values,
        unlearn_values,
        x_label="Representation Similarity",
        y_label="Forgetting Score",
        output_path=f"./figure/ripple_representation_{args.layerid}_{args.model}.png",
    )


if __name__ == "__main__":
    main()
