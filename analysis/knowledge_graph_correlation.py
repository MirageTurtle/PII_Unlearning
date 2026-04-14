import argparse

try:
    from .correlation import (
        analyze_correlation,
        extract_series,
        match_mem_scores_by_csv_lookup,
    )
except ImportError:
    from correlation import (
        analyze_correlation,
        extract_series,
        match_mem_scores_by_csv_lookup,
    )


def main():
    parser = argparse.ArgumentParser(description="PageRank Analysis")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="npo",
        help="the model to calculate the forgetting score",
    )
    args = parser.parse_args()

    mem_score_path = f"results/mem_score/unknown_forgotten_log_{args.model}.json"
    ppr_path = "results/knowledge_graph/ppr_similarity.csv"

    matches = match_mem_scores_by_csv_lookup(
        mem_score_path,
        ppr_path,
        key_column="email",
        metric_fields=["ppr_score"],
    )

    ppr_values, unlearn_values = extract_series(matches, "ppr_score")
    assert len(ppr_values) == len(unlearn_values)

    analyze_correlation(
        ppr_values,
        unlearn_values,
        x_label="PageRank Score",
        y_label="Forgetting Score",
        output_path=f"./figure/ripple_ppr_{args.model}.png",
    )


if __name__ == "__main__":
    main()
