"""
final_clickbait_classifier.py

Final packaging script for the three-class clickbait system.

What it does:
1. Reads a CSV containing predicted regression scores
2. Applies the three-class threshold logic
3. Writes a clean final predictions file

Expected input:
- A CSV with at least a 'predicted_score' column
- Optional useful columns:
  postText_clean, targetTitle_clean, targetDescription, truthMean, true_label

Example:
    python final_clickbait_classifier.py \
        --input clickbait_predictions_transformer_v2.csv \
        --output final_clickbait_predictions.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from threshold_utils import ThresholdConfig, add_three_class_columns


DEFAULT_INPUT = "clickbait_predictions_transformer_v2.csv"
DEFAULT_OUTPUT = "final_clickbait_predictions.csv"


def build_final_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the output readable and GitHub/demo friendly.
    """
    preferred_order = [
        "id",
        "postText_clean",
        "targetTitle_clean",
        "targetDescription",
        "predicted_score",
        "three_class_pred",
        "three_class_label",
        "confidence_band",
        "prediction_rationale",
        "true_label",
        "truthMean",
        "predicted",
        "cross_entropy_loss",
    ]

    existing_cols = [col for col in preferred_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    return df[existing_cols + remaining_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply final three-class clickbait labeling.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV with predicted_score column.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save final output CSV.")
    parser.add_argument("--low", type=float, default=0.25, help="Low threshold for not_clickbait.")
    parser.add_argument("--high", type=float, default=0.60, help="High threshold for clickbait.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, encoding="latin-1")
    config = ThresholdConfig(low=args.low, high=args.high)

    final_df = add_three_class_columns(df, score_col="predicted_score", config=config)
    final_df = build_final_output(final_df)

    final_df.to_csv(args.output, index=False)

    print("=" * 70)
    print("FINAL CLICKBait CLASSIFIER")
    print("=" * 70)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Thresholds:  low={args.low:.2f}, high={args.high:.2f}")
    print()
    print("Prediction distribution:")
    print(final_df["three_class_label"].value_counts(dropna=False).to_string())
    print()
    print("Saved final predictions successfully.")


if __name__ == "__main__":
    main()