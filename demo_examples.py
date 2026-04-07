"""
demo_examples.py

Small demo script for showing representative examples from:
- not_clickbait
- ambiguous
- clickbait

Uses the final three-class predictions file if available,
otherwise falls back to three_class_predictions.csv.

Example:
    python demo_examples.py
"""

from __future__ import annotations

import os

import pandas as pd


PREFERRED_FILES = [
    "final_clickbait_predictions.csv",
    "three_class_predictions.csv",
]


def pick_input_file() -> str:
    for file_name in PREFERRED_FILES:
        if os.path.exists(file_name):
            return file_name
    raise FileNotFoundError(
        "Could not find final_clickbait_predictions.csv or three_class_predictions.csv."
    )


def shorten(text: str, max_len: int = 180) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_examples_for_label(df: pd.DataFrame, label: str, n: int = 3) -> None:
    subset = df[df["three_class_label"] == label].copy()

    if subset.empty:
        print(f"\nNo examples found for: {label}")
        return

    if "predicted_score" in subset.columns:
        if label == "not_clickbait":
            subset = subset.sort_values("predicted_score", ascending=True)
        elif label == "clickbait":
            subset = subset.sort_values("predicted_score", ascending=False)
        else:
            subset["distance_to_middle"] = (subset["predicted_score"] - 0.425).abs()
            subset = subset.sort_values("distance_to_middle", ascending=True)

    print("\n" + "=" * 70)
    print(f"{label.upper()} EXAMPLES")
    print("=" * 70)

    for i, (_, row) in enumerate(subset.head(n).iterrows(), start=1):
        print(f"\nExample {i}")
        print(f"Score: {row.get('predicted_score', 'N/A')}")
        print(f"truthMean: {row.get('truthMean', 'N/A')}")
        print(f"true_label: {row.get('true_label', 'N/A')}")
        print(f"Text: {shorten(row.get('postText_clean', row.get('postText', '')))}")
        if "targetTitle_clean" in row:
            print(f"Title: {shorten(row.get('targetTitle_clean', ''))}")
        elif "targetTitle" in row:
            print(f"Title: {shorten(row.get('targetTitle', ''))}")
        if "prediction_rationale" in row:
            print(f"Why: {row['prediction_rationale']}")


def main() -> None:
    input_file = pick_input_file()
    df = pd.read_csv(input_file, encoding="latin-1")

    required = {"three_class_label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    print(f"Using file: {input_file}")
    print_examples_for_label(df, "not_clickbait", n=3)
    print_examples_for_label(df, "ambiguous", n=3)
    print_examples_for_label(df, "clickbait", n=3)


if __name__ == "__main__":
    main()