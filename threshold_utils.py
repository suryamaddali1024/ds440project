"""
threshold_utils.py

Utility functions for converting regression scores into:
- not_clickbait
- ambiguous
- clickbait

Default thresholds come from the three-class analysis script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ThresholdConfig:
    low: float = 0.25
    high: float = 0.60


LABEL_NOT_CLICKBAIT = "not_clickbait"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_CLICKBAIT = "clickbait"

LABEL_TO_INT = {
    LABEL_NOT_CLICKBAIT: 0,
    LABEL_AMBIGUOUS: 1,
    LABEL_CLICKBAIT: 2,
}

INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}


def validate_thresholds(config: ThresholdConfig) -> None:
    """Validate threshold ordering and range."""
    if not (0.0 <= config.low <= 1.0 and 0.0 <= config.high <= 1.0):
        raise ValueError("Thresholds must be between 0 and 1.")
    if config.low >= config.high:
        raise ValueError("low threshold must be less than high threshold.")


def score_to_label(score: float, config: ThresholdConfig | None = None) -> str:
    """Convert a regression score into a 3-class label."""
    if config is None:
        config = ThresholdConfig()

    validate_thresholds(config)

    if score < config.low:
        return LABEL_NOT_CLICKBAIT
    if score >= config.high:
        return LABEL_CLICKBAIT
    return LABEL_AMBIGUOUS


def score_to_int(score: float, config: ThresholdConfig | None = None) -> int:
    """Convert a regression score into a 3-class integer code."""
    label = score_to_label(score, config)
    return LABEL_TO_INT[label]


def confidence_band(score: float, config: ThresholdConfig | None = None) -> str:
    """
    Simple confidence-style description based on distance from thresholds.
    This is not model probability calibration, just a user-friendly bucket.
    """
    if config is None:
        config = ThresholdConfig()

    validate_thresholds(config)

    if score < config.low:
        margin = config.low - score
        if margin >= 0.20:
            return "high_confidence_not_clickbait"
        if margin >= 0.10:
            return "moderate_confidence_not_clickbait"
        return "low_confidence_not_clickbait"

    if score >= config.high:
        margin = score - config.high
        if margin >= 0.20:
            return "high_confidence_clickbait"
        if margin >= 0.10:
            return "moderate_confidence_clickbait"
        return "low_confidence_clickbait"

    return "ambiguous"


def simple_rationale(
    score: float,
    truth_mean: Optional[float] = None,
    config: ThresholdConfig | None = None,
) -> str:
    """
    Generate a short explanation string for demos / outputs.
    """
    if config is None:
        config = ThresholdConfig()

    label = score_to_label(score, config)

    if label == LABEL_NOT_CLICKBAIT:
        reason = (
            f"Score {score:.3f} is below the low threshold ({config.low:.2f}), "
            "so the model is treating this as not clickbait."
        )
    elif label == LABEL_CLICKBAIT:
        reason = (
            f"Score {score:.3f} is above the high threshold ({config.high:.2f}), "
            "so the model is treating this as clickbait."
        )
    else:
        reason = (
            f"Score {score:.3f} falls between {config.low:.2f} and {config.high:.2f}, "
            "so the model abstains and marks it as ambiguous."
        )

    if truth_mean is not None:
        reason += f" The annotator consensus score (truthMean) is {truth_mean:.3f}."
    return reason


def add_three_class_columns(
    df: pd.DataFrame,
    score_col: str = "predicted_score",
    config: ThresholdConfig | None = None,
) -> pd.DataFrame:
    """Return a copy of df with three-class prediction columns added."""
    if config is None:
        config = ThresholdConfig()

    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' not found in DataFrame.")

    out = df.copy()
    out["three_class_pred"] = out[score_col].apply(lambda x: score_to_int(float(x), config))
    out["three_class_label"] = out[score_col].apply(lambda x: score_to_label(float(x), config))
    out["confidence_band"] = out[score_col].apply(lambda x: confidence_band(float(x), config))

    if "truthMean" in out.columns:
        out["prediction_rationale"] = out.apply(
            lambda row: simple_rationale(
                float(row[score_col]),
                truth_mean=float(row["truthMean"]),
                config=config,
            ),
            axis=1,
        )
    else:
        out["prediction_rationale"] = out[score_col].apply(
            lambda x: simple_rationale(float(x), truth_mean=None, config=config)
        )

    return out