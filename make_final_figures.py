from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["axes.facecolor"] = "white"

ERROR_FILE = "error_analysis_report.csv"
PREDICTION_CANDIDATES = [
    "results/combined_predictions.csv",
    "results/distilbert_3class_predictions.csv",
    "final_clickbait_predictions.csv",
    "three_class_predictions.csv",
]
FIGURE_DIR = "figures"

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.60

# consistent colors across paper
COLOR_NOT = "#A6CEE3"   # blue
COLOR_AMBIG = "#DD8452" # orange
COLOR_CB = "#55A868"    # green
COLOR_SINGLE = "#B07AA1"   # purple for single-color plots


def get_prediction_file() -> str:
    for file_name in PREDICTION_CANDIDATES:
        if os.path.exists(file_name):
            return file_name
    raise FileNotFoundError(
        "Could not find a valid prediction file in the expected locations."
    )


def ensure_figure_dir() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)


def get_score_col(pred_df: pd.DataFrame) -> str | None:
    if "predicted_score" in pred_df.columns:
        return "predicted_score"
    return None


def get_label_col(pred_df: pd.DataFrame) -> str:
    if "predicted_label" in pred_df.columns:
        return "predicted_label"
    if "three_class_label" in pred_df.columns:
        return "three_class_label"
    if "label" in pred_df.columns:
        return "label"
    raise KeyError(
        "Could not find a label column. Expected one of: "
        "'predicted_label', 'three_class_label', or 'label'."
    )


def save_error_type_counts(error_df: pd.DataFrame) -> None:
    if "error_type" not in error_df.columns:
        print("Skipping error_type_counts because 'error_type' column is missing.")
        return

    counts = error_df["error_type"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=COLOR_SINGLE)
    plt.title("Error Type Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Error Type", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "error_type_counts.png"), dpi=300)
    plt.close()


def save_three_class_distribution(pred_df: pd.DataFrame) -> None:
    label_col = get_label_col(pred_df)

    counts = pred_df[label_col].value_counts().reindex(
        ["not_clickbait", "ambiguous", "clickbait"], fill_value=0
    )

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar", color=[COLOR_NOT, COLOR_AMBIG, COLOR_CB])
    plt.title("Three-Class Prediction Distribution", fontsize=13, fontweight="bold")
    plt.ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "three_class_distribution.png"), dpi=300)
    plt.close()


def save_truthmean_boxplot(pred_df: pd.DataFrame) -> None:
    label_col = get_label_col(pred_df)

    needed = {label_col, "truthMean"}
    if not needed.issubset(pred_df.columns):
        print("Skipping truthMean boxplot because required columns are missing.")
        return

    ordered_labels = ["not_clickbait", "ambiguous", "clickbait"]
    data = [
        pred_df.loc[pred_df[label_col] == label, "truthMean"].dropna()
        for label in ordered_labels
    ]

    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, tick_labels=ordered_labels, patch_artist=True)

    box_colors = [COLOR_NOT, COLOR_AMBIG, COLOR_CB]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    plt.title("truthMean by Predicted Three-Class Label", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("truthMean", fontsize=11)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "truthmean_by_predicted_class.png"), dpi=300)
    plt.close()


def save_score_histogram(pred_df: pd.DataFrame) -> None:
    score_col = get_score_col(pred_df)
    if score_col is None:
        print("Skipping score histogram because predicted_score is missing.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(pred_df[score_col].dropna(), bins=30, color=COLOR_SINGLE, edgecolor="white")
    plt.axvline(LOW_THRESHOLD, linestyle="--", linewidth=2, label=f"low={LOW_THRESHOLD:.2f}")
    plt.axvline(HIGH_THRESHOLD, linestyle="--", linewidth=2, label=f"high={HIGH_THRESHOLD:.2f}")
    plt.title("Predicted Score Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Score", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
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