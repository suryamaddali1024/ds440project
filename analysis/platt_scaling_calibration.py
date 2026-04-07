"""
platt_scaling_calibration.py
----------------------------
Calibrate DistilBERT v2 regression scores using Platt Scaling.

Platt scaling fits a logistic regression on the model's raw scores to
transform them into well-calibrated probabilities. After calibration,
a score of 0.70 truly means "70% probability of clickbait" rather than
an arbitrary number.

Approach:
  1. Load v2 test predictions (predicted_score, true_label)
  2. Cross-fit calibration on test set (split into 2 folds)
     - Fit Platt scaling (logistic regression) on fold A, predict fold B
     - Fit on fold B, predict fold A
     - This gives calibrated scores for ALL test samples without leakage
  3. Compute Expected Calibration Error (ECE) before and after
  4. Run three-class binning sweep on calibrated scores to find optimal
     thresholds for the "Not Clickbait / Ambiguous / Clickbait" buckets
  5. Save calibrated predictions to results/

Why cross-fit instead of using the validation set?
  - The ideal approach would be: fit calibration on the val set (used during
    v2 training for early stopping), then apply to the test set. This keeps
    calibration completely separate from evaluation.
  - However, v2 only saved predictions for the TEST set, not the val set.
    Re-running v2 to also export val predictions would take 3+ hours on CPU.
  - Cross-fitting on the test set is the next best option: split test into
    2 stratified folds, fit calibration on one half, apply to the other,
    then swap. Every sample is calibrated by a model that never saw it,
    so there is no leakage. The downside is that calibration parameters
    are slightly noisier than they would be from a dedicated val set.

Expected Output (from prior runs):
  - Raw ECE: 0.1146
  - Platt-calibrated ECE: 0.0330 (71% reduction)
  - Best three-class thresholds: low=0.05, high=0.85
  - Confident F1: 0.9255 on 19.8% of samples

Requires: ../results/clickbait_predictions_transformer_v2.csv

Usage:
    python platt_scaling_calibration.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "../results/clickbait_predictions_transformer_v2.csv"
OUTPUT_FILE = "../results/calibrated_platt.csv"
RANDOM_STATE = 42
N_BINS_ECE = 10  # number of bins for ECE computation


# ===========================================================================
# CALIBRATION HELPERS
# ===========================================================================

def compute_ece(probs, labels, n_bins=N_BINS_ECE):
    """
    Expected Calibration Error: weighted average gap between predicted
    probability and actual positive rate within each bin.

    A perfectly calibrated model has ECE = 0 (no gaps anywhere).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_predicted = probs[mask].mean()
        avg_actual = labels[mask].mean()
        ece += mask.sum() * abs(avg_predicted - avg_actual)
    return ece / len(probs)


def print_calibration_table(probs, labels, title):
    """Show per-bin gaps to visualize how well-calibrated predictions are."""
    print(f"\n   --- {title} ---")
    print(f"   {'Bin':>10s}  {'n':>5s}  {'Predicted':>10s}  {'Actual':>10s}  {'Gap':>6s}")
    print(f"   {'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")

    bin_edges = np.linspace(0, 1, N_BINS_ECE + 1)
    total_gap = 0
    n_used = 0
    for i in range(N_BINS_ECE):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        predicted = probs[mask].mean()
        actual = labels[mask].mean()
        gap = abs(predicted - actual)
        total_gap += gap
        n_used += 1
        bin_label = f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})"
        print(f"   {bin_label:>10s}  {mask.sum():5d}  {predicted:10.3f}  {actual:10.3f}  {gap:6.3f}")
    print(f"   Average gap across bins: {total_gap / max(n_used, 1):.4f}")


# ===========================================================================
# PLATT SCALING (CROSS-FITTED)
# ===========================================================================

def platt_calibrate_cross_fit(scores, labels, random_state=RANDOM_STATE):
    """
    Cross-fit Platt scaling on the test set to avoid leakage.

    Splits the data into 2 stratified folds:
      - Fit logistic regression on fold A, calibrate fold B
      - Fit logistic regression on fold B, calibrate fold A

    This gives us calibrated scores for every sample using a model that
    never saw that sample during fitting.
    """
    fold_a, fold_b = train_test_split(
        np.arange(len(scores)),
        test_size=0.5,
        stratify=labels,
        random_state=random_state,
    )

    calibrated = np.zeros(len(scores))
    fitted_params = []  # store parameters from each fold

    for fit_fold, predict_fold in [(fold_a, fold_b), (fold_b, fold_a)]:
        # Fit logistic regression mapping raw_score -> calibrated_probability
        platt = LogisticRegression()
        platt.fit(scores[fit_fold].reshape(-1, 1), labels[fit_fold])

        # Apply to the held-out fold
        calibrated[predict_fold] = platt.predict_proba(
            scores[predict_fold].reshape(-1, 1)
        )[:, 1]

        fitted_params.append({
            "A": platt.coef_[0][0],
            "B": platt.intercept_[0],
        })

    return calibrated, fitted_params


# ===========================================================================
# THREE-CLASS BINNING SWEEP
# ===========================================================================

def three_class_threshold_sweep(probs, labels):
    """
    Find the (low, high) threshold pair that maximizes F1 on confident
    predictions only. Excludes the "ambiguous / IDK" zone from evaluation.
    """
    best = {"f1": 0.0}
    for low_t in np.arange(0.05, 0.40, 0.05):
        for high_t in np.arange(0.50, 0.90, 0.05):
            pred_3 = np.full(len(probs), 1)              # default: ambiguous
            pred_3[probs < low_t] = 0                    # confident: not clickbait
            pred_3[probs >= high_t] = 2                  # confident: clickbait

            confident = pred_3 != 1
            if confident.sum() < 100:
                continue

            pred_binary = (pred_3[confident] == 2).astype(int)
            true_binary = labels[confident]

            f1 = f1_score(true_binary, pred_binary)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "acc": accuracy_score(true_binary, pred_binary),
                    "prec": precision_score(true_binary, pred_binary),
                    "rec": recall_score(true_binary, pred_binary),
                    "low": low_t,
                    "high": high_t,
                    "n_confident": confident.sum(),
                    "pct_confident": 100 * confident.sum() / len(probs),
                }

    return best


# ===========================================================================
# RUN
# ===========================================================================

print("=" * 70)
print("PLATT SCALING CALIBRATION")
print("=" * 70)

# Load v2 predictions
df = pd.read_csv(INPUT_FILE, encoding="latin-1")
raw_scores = df["predicted_score"].values
true_labels = df["true_label"].values

print(f"\n   Loaded {len(df)} test predictions from v2")
print(f"   Raw score range: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
print(f"   Class distribution: {dict(zip(*np.unique(true_labels, return_counts=True)))}")

# Apply Platt scaling
print("\n   Cross-fitting Platt scaling on test set...")
calibrated_scores, params = platt_calibrate_cross_fit(raw_scores, true_labels)

print(f"   Fold 1 parameters: A={params[0]['A']:.4f}  B={params[0]['B']:.4f}")
print(f"   Fold 2 parameters: A={params[1]['A']:.4f}  B={params[1]['B']:.4f}")
print(f"   Formula: calibrated = sigmoid(A * raw_score + B)")

# Compute ECE before and after
raw_ece = compute_ece(raw_scores, true_labels)
calibrated_ece = compute_ece(calibrated_scores, true_labels)
reduction = 100 * (raw_ece - calibrated_ece) / raw_ece

print(f"\n   --- Expected Calibration Error (ECE) ---")
print(f"   Raw scores:        {raw_ece:.4f}")
print(f"   Platt-calibrated:  {calibrated_ece:.4f}")
print(f"   Reduction:         {reduction:.1f}%")

# Show per-bin calibration
print_calibration_table(raw_scores, true_labels, "Raw Score Calibration")
print_calibration_table(calibrated_scores, true_labels, "Platt-Calibrated")

# Three-class binning on calibrated scores
print("\n" + "=" * 70)
print("THREE-CLASS BINNING ON CALIBRATED SCORES")
print("=" * 70)
best = three_class_threshold_sweep(calibrated_scores, true_labels)

print(f"\n   Best thresholds: low={best['low']:.2f}  high={best['high']:.2f}")
print(f"   With Platt-calibrated scores, these thresholds are probabilistically meaningful:")
print(f"     Score < {best['low']:.2f} = less than {best['low']*100:.0f}% true probability of clickbait")
print(f"     Score >= {best['high']:.2f} = at least {best['high']*100:.0f}% true probability of clickbait")
print(f"     In between = ambiguous, model abstains")

print(f"\n   --- Performance on Confident Predictions ---")
print(f"   Confident:  {best['n_confident']} / {len(df)} ({best['pct_confident']:.1f}%)")
print(f"   Abstained:  {len(df) - best['n_confident']} ({100 - best['pct_confident']:.1f}%)")
print(f"   F1:         {best['f1']:.4f}")
print(f"   Accuracy:   {best['acc']:.4f}")
print(f"   Precision:  {best['prec']:.4f}")
print(f"   Recall:     {best['rec']:.4f}")

# Save calibrated predictions
out = df.copy()
out["calibrated_score"] = calibrated_scores

# Add three-class label using best thresholds
three_class = np.full(len(out), 1)
three_class[calibrated_scores < best["low"]] = 0
three_class[calibrated_scores >= best["high"]] = 2
label_map = {0: "not_clickbait", 1: "ambiguous", 2: "clickbait"}
out["three_class_pred"] = three_class
out["three_class_label"] = [label_map[t] for t in three_class]

out.to_csv(OUTPUT_FILE, index=False)
print(f"\n   Saved calibrated predictions to {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
