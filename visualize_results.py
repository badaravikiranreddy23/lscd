"""
visualize_results.py
Plots graded change scores for all words as a bar chart.
Saves to outputs/change_scores_plot.png

Run: python visualize_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


HIGH_CHANGE_EXPECTED = {"mask", "variant", "lockdown", "essential", "bubble"}


def plot_from_csv(csv_path: str = "outputs/scores_summary.csv",
                  output_path: str = "outputs/change_scores_plot.png") -> None:
    """Plot change scores from CSV summary."""
    import csv

    words, scores, colors = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            words.append(row["word"])
            scores.append(float(row["graded_score"]))
            colors.append("#e74c3c" if row["expected_change"] == "high" else "#3498db")

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    words   = [words[i] for i in order]
    scores  = [scores[i] for i in order]
    colors  = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(words, scores, color=colors, edgecolor="white", linewidth=0.8)

    # Threshold line
    ax.axhline(y=0.3, color="gray", linestyle="--", linewidth=1.2, label="Low/Moderate boundary (0.3)")
    ax.axhline(y=0.6, color="black", linestyle="--", linewidth=1.2, label="Moderate/High boundary (0.6)")

    # Labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    # Legend
    high_patch = mpatches.Patch(color="#e74c3c", label="Expected HIGH change (pandemic words)")
    low_patch  = mpatches.Patch(color="#3498db", label="Expected LOW change (control words)")
    ax.legend(handles=[high_patch, low_patch], loc="upper right", fontsize=9)

    ax.set_title("Social Media Semantic Change Scores\n(Pre-2020 vs Post-2020 Reddit Corpus)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Target Word", fontsize=11)
    ax.set_ylabel("Graded Change Score (JSD)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {output_path}")
    plt.close()


def plot_sense_shift_summary(report_path: str = "outputs/full_report.json",
                              output_path: str = "outputs/sense_shift_summary.png") -> None:
    """
    For each word, show number of senses in C1 vs C2
    to visualize sense gain/loss.
    """
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}. Run social_analyzer.py --all first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    words, n_old, n_new = [], [], []
    for word, data in report.items():
        words.append(word)
        n_old.append(len(data.get("old_period_senses", {})))
        n_new.append(len(data.get("new_period_senses", {})))

    x = np.arange(len(words))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, n_old, width, label="Pre-2020 Senses", color="#95a5a6")
    ax.bar(x + width/2, n_new, width, label="Post-2020 Senses", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylabel("Number of Sense Clusters")
    ax.set_title("Sense Cluster Count: Pre vs Post 2020", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Sense shift plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    if os.path.exists("outputs/scores_summary.csv"):
        plot_from_csv()
    else:
        print("Run social_analyzer.py --all first to generate outputs/scores_summary.csv")

    if os.path.exists("outputs/full_report.json"):
        plot_sense_shift_summary()
