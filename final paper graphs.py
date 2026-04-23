"""
make_paper_figures.py

Creates 9 clean figures for the final clickbait paper.


What this script does:
- Generates final paper-ready figures in a separate output folder
- Uses your saved CSVs where available
- Falls back to manually-entered values for model summary figures

You may need to edit only the CONFIG section below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =========================================================
# CONFIG: EDIT THESE PATHS / VALUES IF NEEDED
# =========================================================

PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "final_paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Main predictions file for modern headlines
MODERN_PREDICTIONS_CSV = RESULTS_DIR / "combined_predictions.csv"

# Optional file for error breakdown
ERROR_BREAKDOWN_CSV = RESULTS_DIR / "error_breakdown.csv"

# Optional file for confident-vs-coverage tradeoff
CONFIDENCE_SUMMARY_CSV = RESULTS_DIR / "confidence_tradeoff_summary.csv"

# Optional file for feature ablation
ABLATION_CSV = RESULTS_DIR / "ablation_results.csv"

# Optional file for calibration
# If you already have a file with y_true / y_prob_before / y_prob_after, set it here
CALIBRATION_CSV = RESULTS_DIR / "calibration_data.csv"

# If you do not have summary CSVs for some plots, these hardcoded values will be used
MODEL_PERFORMANCE = {
    "Logistic Regression": 0.615,
    "LightGBM": 0.611,
    "DistilBERT v1": 0.697,
    "DistilBERT v2": 0.708,
    "Ensemble": 0.711,
    "RoBERTa": 0.708,
    "Final 3-Class (Confident)": 0.87 
}

CONFIDENT_VS_COVERAGE = {
    "DistilBERT v2": {"confident_f1": 0.835, "coverage": 0.79},
    "Ensemble": {"confident_f1": 0.847, "coverage": 0.76},
    "3-Class Final": {"confident_f1": 0.870, "coverage": 0.72},
}

MODEL_PROGRESSION = {
    "Feature-based": 0.615,
    "DistilBERT": 0.708,
    "Ensemble": 0.711,
    "3-Class Confident": 0.870,
}

ERROR_BREAKDOWN = {
    "Genuine Errors": 52.7,
    "Ambiguous": 47.3,
}

# Feature ablation fallback if CSV not found
FEATURE_ABLATION = {
    "Full Features": 0.615,
    "Without Metadata": 0.608,
    "Without Sentiment": 0.610,
    "Without Linguistic": 0.611,
    "Without Error-driven": 0.608,
}

# Class order for all plots
CLASS_ORDER = ["not_clickbait", "ambiguous", "clickbait"]

# Clean, consistent colors for paper
CLASS_COLORS = {
    "not_clickbait": "#A6CEE3",   # muted blue
    "ambiguous": "#DDC44A",       # muted gold
    "clickbait": "#C44E52",       # muted red
}

MODEL_BAR_COLOR = "#A6CEE3"
LINE_COLOR = "#A6CEE3"
SECONDARY_COLOR = "#C44E52"
NEUTRAL_COLOR = "#7F7F7F"

DPI = 300
FIGSIZE_WIDE = (9, 5.5)
FIGSIZE_STANDARD = (7.5, 5.2)
FIGSIZE_TALL = (8, 6)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
})


# =========================================================
# HELPERS
# =========================================================

def save_figure(filename: str) -> None:
    out_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    print(f"Warning: file not found -> {path}")
    return None


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "non_clickbait": "not_clickbait",
        "not clickbait": "not_clickbait",
        "not-clickbait": "not_clickbait",
        "not_clickbait": "not_clickbait",
        "genuine": "not_clickbait",
        "ambiguous": "ambiguous",
        "uncertain": "ambiguous",
        "borderline": "ambiguous",
        "clickbait": "clickbait",
    }
    return mapping.get(text, text)


def find_first_matching_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def get_prediction_label_column(df: pd.DataFrame) -> Optional[str]:
    return find_first_matching_column(df, [
        "predicted_label",
        "prediction",
        "pred_label",
        "label",
        "final_label",
        "predicted_class",
    ])


def get_source_column(df: pd.DataFrame) -> Optional[str]:
    return find_first_matching_column(df, [
        "source",
        "publisher",
        "outlet",
        "news_source",
    ])


def get_confidence_column(df: pd.DataFrame) -> Optional[str]:
    # priority: direct confidence-like fields
    direct = find_first_matching_column(df, [
        "confidence",
        "conf_score",
        "max_probability",
        "max_prob",
        "pred_confidence",
        "score",
        "probability",
        "predicted_probability",
    ])
    if direct:
        return direct

    # otherwise infer from multiple probability columns
    prob_cols = []
    for col in df.columns:
        c = col.lower()
        if any(token in c for token in ["prob", "score"]):
            prob_cols.append(col)

    if prob_cols:
        # create max confidence column
        df["_derived_confidence"] = df[prob_cols].max(axis=1)
        return "_derived_confidence"

    return None


def add_bar_labels(ax, fmt: str = "{:.3f}", percent: bool = False) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        if percent:
            label = f"{height:.1f}%"
        else:
            label = fmt.format(height)
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points"
        )


def clean_source_name(x: str) -> str:
    x = str(x).strip()
    replacements = {
        "bbc news": "BBC",
        "bbc": "BBC",
        "npr.org": "NPR",
        "npr": "NPR",
        "fox news": "Fox",
        "foxnews": "Fox",
        "buzzfeed": "BuzzFeed",
        "cnn": "CNN",
        "new york times": "NYT",
        "the new york times": "NYT",
        "washington post": "Washington Post",
    }
    key = x.lower()
    return replacements.get(key, x)


# =========================================================
# FIGURE 1: MODEL PERFORMANCE COMPARISON
# =========================================================

def plot_model_performance_comparison() -> None:
    df = pd.DataFrame({
        "Model": list(MODEL_PERFORMANCE.keys()),
        "F1 Score": list(MODEL_PERFORMANCE.values()),
    })

    plt.figure(figsize=FIGSIZE_WIDE)
    bars = plt.bar(df["Model"], df["F1 Score"], color=MODEL_BAR_COLOR, edgecolor="black", linewidth=0.6)
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.title("Model Performance Comparison (Binary vs. Three-Class Framework")
    plt.ylim(0.55, max(df["F1 Score"]) + 0.05)
    plt.xticks(rotation=20, ha="right")

    ax = plt.gca()
    add_bar_labels(ax, fmt="{:.3f}")

    save_figure("figure_1_model_performance_comparison.png")


# =========================================================
# FIGURE 2: CONFIDENT VS COVERAGE TRADEOFF
# =========================================================

def plot_confident_vs_coverage_tradeoff() -> None:
    df = read_csv_if_exists(CONFIDENCE_SUMMARY_CSV)

    if df is None:
        df = pd.DataFrame({
            "Model": list(CONFIDENT_VS_COVERAGE.keys()),
            "Confident F1": [v["confident_f1"] for v in CONFIDENT_VS_COVERAGE.values()],
            "Coverage": [v["coverage"] for v in CONFIDENT_VS_COVERAGE.values()],
        })
    else:
        # try to standardize expected column names
        model_col = find_first_matching_column(df, ["model", "approach"])
        f1_col = find_first_matching_column(df, ["confident_f1", "f1_confident", "conf_f1"])
        coverage_col = find_first_matching_column(df, ["coverage", "confident_pct", "percent_confident", "% confident"])

        if not all([model_col, f1_col, coverage_col]):
            raise ValueError("confidence_tradeoff_summary.csv is missing expected columns.")

        df = df[[model_col, f1_col, coverage_col]].copy()
        df.columns = ["Model", "Confident F1", "Coverage"]

    x = np.arange(len(df))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2,
        df["Confident F1"],
        width,
        label="Confident F1",
        color=MODEL_BAR_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    bars2 = ax2.bar(
        x + width / 2,
        df["Coverage"] * 100,
        width,
        label="% Confident",
        color=SECONDARY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    ax1.set_xlabel("Model / Approach")
    ax1.set_ylabel("Confident F1")
    ax2.set_ylabel("% Confident Predictions")
    ax1.set_title("Confident vs Coverage Tradeoff (Three-Class)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax1.set_ylim(0.70, max(df["Confident F1"]) + 0.05)
    ax2.set_ylim(0, 100)

    ax1.legend([bars1, bars2], ["Confident F1", "% Confident"], loc="upper left")

    save_figure("figure_2_confident_vs_coverage_tradeoff.png")


# =========================================================
# FIGURE 3: MODERN PREDICTION DISTRIBUTION
# =========================================================

def plot_modern_prediction_distribution(pred_df: pd.DataFrame) -> None:
    label_col = get_prediction_label_column(pred_df)
    if label_col is None:
        raise ValueError("Could not find a prediction label column in modern predictions file.")

    labels = pred_df[label_col].map(normalize_label)
    counts = labels.value_counts().reindex(CLASS_ORDER, fill_value=0)
    percents = 100 * counts / counts.sum()

    plt.figure(figsize=FIGSIZE_STANDARD)
    bars = plt.bar(
        counts.index,
        percents.values,
        color=[CLASS_COLORS[c] for c in counts.index],
        edgecolor="black",
        linewidth=0.6,
    )
    plt.ylabel("Percentage of Predictions")
    plt.xlabel("Predicted Class")
    plt.title("Model Prediction Distribution on Modern Headlines")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())

    ax = plt.gca()
    add_bar_labels(ax, percent=True)

    save_figure("figure_3_modern_prediction_distribution.png")


# =========================================================
# FIGURE 4: SOURCE-WISE STACKED BREAKDOWN
# =========================================================

def plot_source_wise_breakdown(pred_df: pd.DataFrame) -> None:
    label_col = get_prediction_label_column(pred_df)
    source_col = get_source_column(pred_df)

    if label_col is None:
        raise ValueError("Could not find a prediction label column for source-wise breakdown.")
    if source_col is None:
        raise ValueError("Could not find a source column for source-wise breakdown.")

    temp = pred_df[[source_col, label_col]].copy()
    temp[source_col] = temp[source_col].map(clean_source_name)
    temp[label_col] = temp[label_col].map(normalize_label)

    # Count total articles per source
    source_counts = temp[source_col].value_counts()

    # Keep only top 12 sources
    top_sources = source_counts.head(12).index
    temp = temp[temp[source_col].isin(top_sources)].copy()

    # Build source x class table
    ctab = pd.crosstab(temp[source_col], temp[label_col]).reindex(columns=CLASS_ORDER, fill_value=0)

    # Convert to percentages within each source
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100

    # Optional: sort by clickbait percentage descending
    if "clickbait" in ctab_pct.columns:
        ctab_pct = ctab_pct.sort_values(by="clickbait", ascending=True)
    else:
        ctab_pct = ctab_pct.sort_index()

    plt.figure(figsize=(10, 8))
    left = np.zeros(len(ctab_pct))

    for class_name in CLASS_ORDER:
        values = ctab_pct[class_name].values
        plt.barh(
            ctab_pct.index,
            values,
            left=left,
            label=class_name,
            color=CLASS_COLORS[class_name],
            edgecolor="black",
            linewidth=0.4,
        )
        left += values

    plt.xlabel("Percentage Within Source")
    plt.ylabel("Source")
    plt.title("Source-wise Prediction Breakdown on Modern Headlines")
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.legend(title="Predicted Class", loc="lower right")

    save_figure("figure_4_source_wise_breakdown.png")


# =========================================================
# FIGURE 5: MODEL PROGRESSION LINE CHART
# =========================================================

def plot_model_progression() -> None:
    stages = list(MODEL_PROGRESSION.keys())
    scores = list(MODEL_PROGRESSION.values())

    plt.figure(figsize=FIGSIZE_WIDE)
    plt.plot(
        stages,
        scores,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color=LINE_COLOR,
    )

    for x, y in zip(stages, scores):
        plt.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    plt.ylabel("F1 Score")
    plt.xlabel("Model Stage")
    plt.title("Model Progression Across the Project")
    plt.ylim(0.58, max(scores) + 0.05)

    save_figure("figure_5_model_progression.png")


# =========================================================
# FIGURE 6: ERROR BREAKDOWN
# =========================================================

def plot_error_breakdown() -> None:
    df = read_csv_if_exists(ERROR_BREAKDOWN_CSV)

    if df is None:
        data = ERROR_BREAKDOWN
    else:
        category_col = find_first_matching_column(df, ["category", "error_type", "group", "label"])
        value_col = find_first_matching_column(df, ["percent", "percentage", "value", "count"])

        if not all([category_col, value_col]):
            raise ValueError("error_breakdown.csv is missing expected columns.")

        data = dict(zip(df[category_col], df[value_col]))

    categories = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=FIGSIZE_STANDARD)
    bars = plt.bar(
        categories,
        values,
        color=[SECONDARY_COLOR, "#E6B94C"],
        edgecolor="black",
        linewidth=0.6,
    )
    plt.ylabel("Percentage")
    plt.xlabel("Error Category")
    plt.title("Error Breakdown")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())

    ax = plt.gca()
    add_bar_labels(ax, percent=True)

    save_figure("figure_6_error_breakdown.png")


# =========================================================
# FIGURE 7: CALIBRATION PLOT
# =========================================================

def calibration_curve_manual(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    prob_true = []
    prob_pred = []

    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        prob_true.append(y_true[mask].mean())
        prob_pred.append(y_prob[mask].mean())

    return np.array(prob_pred), np.array(prob_true)


def plot_calibration() -> None:
    df = read_csv_if_exists(CALIBRATION_CSV)
    if df is None:
        print("Skipping calibration plot because calibration_data.csv was not found.")
        return

    true_col = find_first_matching_column(df, ["y_true", "truth", "label", "actual"])
    before_col = find_first_matching_column(df, ["y_prob_before", "prob_before", "uncalibrated_prob"])
    after_col = find_first_matching_column(df, ["y_prob_after", "prob_after", "calibrated_prob"])

    if true_col is None or before_col is None:
        print("Skipping calibration plot because required columns were not found.")
        return

    y_true = df[true_col].to_numpy()
    y_before = df[before_col].to_numpy()

    plt.figure(figsize=FIGSIZE_STANDARD)
    x_before, y_before_curve = calibration_curve_manual(y_true, y_before, n_bins=10)
    plt.plot(x_before, y_before_curve, marker="o", linewidth=2, label="Before Calibration")

    if after_col is not None:
        y_after = df[after_col].to_numpy()
        x_after, y_after_curve = calibration_curve_manual(y_true, y_after, n_bins=10)
        plt.plot(x_after, y_after_curve, marker="o", linewidth=2, label="After Isotonic")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=NEUTRAL_COLOR, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Calibration Plot")
    plt.legend()

    save_figure("figure_7_calibration_plot.png")


# =========================================================
# FIGURE 8: FEATURE ABLATION IMPACT
# =========================================================

def plot_feature_ablation() -> None:
    df = read_csv_if_exists(ABLATION_CSV)

    if df is None:
        ablation_data = FEATURE_ABLATION
        plot_df = pd.DataFrame({
            "Configuration": list(ablation_data.keys()),
            "F1 Score": list(ablation_data.values()),
        })
    else:
        config_col = find_first_matching_column(df, [
            "feature_set",
            "configuration",
            "ablation",
            "model",
            "setup",
        ])
        f1_col = find_first_matching_column(df, ["f1", "f1_score", "test_f1", "val_f1"])

        if not all([config_col, f1_col]):
            raise ValueError("ablation_results.csv is missing expected columns.")

        plot_df = df[[config_col, f1_col]].copy()
        plot_df.columns = ["Configuration", "F1 Score"]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(
        plot_df["Configuration"],
        plot_df["F1 Score"],
        color=MODEL_BAR_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    plt.ylabel("F1 Score")
    plt.xlabel("Feature Configuration")
    plt.title("Feature Ablation Impact")
    plt.ylim(min(plot_df["F1 Score"]) - 0.02, max(plot_df["F1 Score"]) + 0.02)
    plt.xticks(rotation=22, ha="right")

    ax = plt.gca()
    add_bar_labels(ax, fmt="{:.3f}")

    save_figure("figure_8_feature_ablation_impact.png")


# =========================================================
# FIGURE 9: CONFIDENCE SCORE HISTOGRAM
# =========================================================

def plot_confidence_score_histogram(pred_df: pd.DataFrame) -> None:
    conf_col = get_confidence_column(pred_df)
    if conf_col is None:
        print("Skipping confidence histogram because no confidence/probability column was found.")
        return

    values = pred_df[conf_col].dropna().astype(float)

    plt.figure(figsize=FIGSIZE_STANDARD)
    plt.hist(
        values,
        bins=20,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        color=MODEL_BAR_COLOR,
    )
    plt.axvline(0.2, linestyle="--", linewidth=1.5, color="#DD8452", label="0.2 Threshold")
    plt.axvline(0.6, linestyle="--", linewidth=1.5, color=SECONDARY_COLOR, label="0.6 Threshold")
    plt.xlabel("Prediction Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Confidence Score Distribution")
    plt.legend()

    save_figure("figure_9_confidence_score_histogram.png")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    pred_df = read_csv_if_exists(MODERN_PREDICTIONS_CSV)

    plot_model_performance_comparison()
    plot_confident_vs_coverage_tradeoff()

    if pred_df is not None:
        plot_modern_prediction_distribution(pred_df)
        plot_source_wise_breakdown(pred_df)
    else:
        print("Skipping Figures 3 and 4 because modern predictions file was not found.")

    plot_model_progression()
    plot_error_breakdown()
    plot_calibration()
    plot_feature_ablation()

    if pred_df is not None:
        plot_confidence_score_histogram(pred_df)
    else:
        print("Skipping Figure 9 because modern predictions file was not found.")

    print("\nDone. Final paper figures are in:")
    print(FIGURES_DIR.resolve())


if __name__ == "__main__":
    main()