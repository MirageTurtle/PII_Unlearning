import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_mem_scores(mem_score_path):
    data = load_json(mem_score_path)
    return [item for item in data if "gt" in item]


def normalize_full_text(full_text):
    if isinstance(full_text, list):
        return full_text[0] if full_text else ""
    return full_text or ""


def match_mem_scores_by_text(mem_score_path, metric_path, metric_fields):
    mem_scores = load_mem_scores(mem_score_path)
    metrics = load_json(metric_path)

    matches = []
    for mem_item in mem_scores:
        gt_value = mem_item["gt"]
        for metric_item in metrics:
            if "full_text" not in metric_item:
                continue

            if gt_value not in normalize_full_text(metric_item["full_text"]):
                continue

            match = {
                "gt": gt_value,
                "unlearn_log": mem_item.get("unlearn_log"),
                "janus_log": mem_item.get("janus_log"),
            }
            for field in metric_fields:
                match[field] = metric_item.get(field)
            matches.append(match)
            print_match(match, metric_fields)
            break

    return matches


def match_mem_scores_by_csv_lookup(mem_score_path, csv_path, key_column, metric_fields):
    mem_scores = load_mem_scores(mem_score_path)
    metric_df = pd.read_csv(csv_path)
    metric_lookup = metric_df.set_index(key_column)[metric_fields].to_dict("index")

    matches = []
    for mem_item in mem_scores:
        gt_value = mem_item["gt"]
        metric_values = metric_lookup.get(gt_value)
        if metric_values is None:
            print(f"Warning: {gt_value} not found in {csv_path}")
            continue

        match = {
            "gt": gt_value,
            "unlearn_log": mem_item.get("unlearn_log"),
            "janus_log": mem_item.get("janus_log"),
        }
        match.update(metric_values)
        matches.append(match)
        print_match(match, metric_fields)

    return matches


def print_match(match, metric_fields):
    print(f"Matched: gt='{match['gt']}'")
    print(f"  unlearn_log: {match.get('unlearn_log')}")
    print(f"  janus_log:   {match.get('janus_log')}")
    for field in metric_fields:
        print(f"  {field}: {match.get(field)}")
    print("-" * 50)


def extract_series(matches, x_field, y_field="unlearn_log"):
    x_values = [match[x_field] for match in matches if match.get(x_field) is not None and match.get(y_field) is not None]
    y_values = [match[y_field] for match in matches if match.get(x_field) is not None and match.get(y_field) is not None]
    return x_values, y_values


def analyze_correlation(
    x_values,
    y_values,
    *,
    x_label,
    y_label,
    output_path,
    x_ticks=None,
    x_ticklabels=None,
):
    x_array = np.array(x_values, dtype=float)
    y_array = np.array(y_values, dtype=float)

    pearson_r, pearson_p = pearsonr(x_array, y_array)
    spearman_r, spearman_p = spearmanr(x_array, y_array)

    print(f"Pearson: {pearson_r:.4f} (p value: {pearson_p:.4f})")
    print(f"Spearman: {spearman_r:.4f} (p value: {spearman_p:.4f})")

    plt.figure(figsize=(8, 8))
    plt.scatter(x_array, y_array, alpha=1.0, s=60, color="#3498db", edgecolors="white")

    trend = np.poly1d(np.polyfit(x_array, y_array, 1))
    plt.plot(x_array, trend(x_array), "b--", alpha=0.9, linewidth=3, label="Trend")

    if x_ticks is not None:
        plt.xticks(x_ticks, x_ticklabels, fontsize=25)
    else:
        plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend()

    text = (
        f"r: {pearson_r:.4f} ($p$-value: {pearson_p:.4f})\n"
        f"ρ: {spearman_r:.4f} ($p$-value: {spearman_p:.4f})"
    )
    plt.text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        fontsize=25,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()

    return {
        "pearson_correlation": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_r,
        "spearman_p_value": spearman_p,
    }
