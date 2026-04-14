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
    mem_score_path = "results/mem_score/unknown_forgotten_log_ga.json"
    gradient_path = "results/gradients/forgotten_unknown_unlearn_ga.json"

    matches = match_mem_scores_by_text(
        mem_score_path,
        gradient_path,
        metric_fields=["cosine_similarity", "dot_product"],
    )

    dot_product_values, unlearn_values = extract_series(matches, "dot_product")
    assert len(dot_product_values) == len(unlearn_values)

    analyze_correlation(
        dot_product_values,
        unlearn_values,
        x_label="Dot Product of Gradients",
        y_label="Forgetting Score",
        output_path="./figure/ripple_gradient_ga.png",
        x_ticks=[-5000, -2500, 0, 2500, 5000, 7500, 10000],
        x_ticklabels=["-5k", "-2.5k", "0k", "2.5k", "5k", "7.5k", "10k"],
    )


if __name__ == "__main__":
    main()
