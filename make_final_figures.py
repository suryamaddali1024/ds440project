"""
make_final_figures.py

Creates final project figures from:
- error_analysis_report.csv
- three_class_predictions.csv or final_clickbait_predictions.csv

Outputs:
- figures/error_type_counts.png
- figures/three_class_distribution.png
- figures/truthmean_by_predicted_class.png
- figures/score_histogram_with_thresholds.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


ERROR_FILE = "error_analysis_report.csv"
PREDICTION_CANDIDATES = [
    "final_clickbait_predictions.csv",
    "three_class_predictions.csv",
]
FIGURE_DIR = "figures"

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.60


def get_prediction_file() -> str:
    for file_name in PREDICTION_CANDIDATES:
        if os.path.exists(file_name):
            return file_name
    raise FileNotFoundError(
        "Could not find final_clickbait_predictions.csv or three_class_predictions.csv."
    )


def ensure_figure_dir() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)


def save_error_type_counts(error_df: pd.DataFrame) -> None:
    counts = error_df["error_type"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Error Type Counts")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "error_type_counts.png"), dpi=300)
    plt.close()


def save_three_class_distribution(pred_df: pd.DataFrame) -> None:
    counts = pred_df["three_class_label"].value_counts().reindex(
        ["not_clickbait", "ambiguous", "clickbait"], fill_value=0
    )

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Three-Class Prediction Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "three_class_distribution.png"), dpi=300)
    plt.close()


def save_truthmean_boxplot(pred_df: pd.DataFrame) -> None:
    needed = {"three_class_label", "truthMean"}
    if not needed.issubset(pred_df.columns):
        print("Skipping truthMean boxplot because required columns are missing.")
        return

    ordered_labels = ["not_clickbait", "ambiguous", "clickbait"]
    data = [
        pred_df.loc[pred_df["three_class_label"] == label, "truthMean"].dropna()
        for label in ordered_labels
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=ordered_labels)
    plt.title("truthMean by Predicted Three-Class Label")
    plt.xlabel("Predicted Class")
    plt.ylabel("truthMean")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "truthmean_by_predicted_class.png"), dpi=300)
    plt.close()


def save_score_histogram(pred_df: pd.DataFrame) -> None:
    if "predicted_score" not in pred_df.columns:
        print("Skipping score histogram because predicted_score is missing.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["predicted_score"].dropna(), bins=30)
    plt.axvline(LOW_THRESHOLD, linestyle="--", linewidth=2, label=f"low={LOW_THRESHOLD:.2f}")
    plt.axvline(HIGH_THRESHOLD, linestyle="--", linewidth=2, label=f"high={HIGH_THRESHOLD:.2f}")
    plt.title("Predicted Score Distribution with Three-Class Thresholds")
    plt.xlabel("Predicted Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "score_histogram_with_thresholds.png"), dpi=300)
    plt.close()


def main() -> None:
    ensure_figure_dir()

    if not os.path.exists(ERROR_FILE):
        raise FileNotFoundError(f"Missing required file: {ERROR_FILE}")

    prediction_file = get_prediction_file()

    error_df = pd.read_csv(ERROR_FILE, encoding="latin-1")
    pred_df = pd.read_csv(prediction_file, encoding="latin-1")

    save_error_type_counts(error_df)
    save_three_class_distribution(pred_df)
    save_truthmean_boxplot(pred_df)
    save_score_histogram(pred_df)

    print("=" * 70)
    print("FINAL FIGURES CREATED")
    print("=" * 70)
    print(f"Used error file:      {ERROR_FILE}")
    print(f"Used prediction file: {prediction_file}")
    print(f"Saved figures to:     {FIGURE_DIR}/")


if __name__ == "__main__":
    main()