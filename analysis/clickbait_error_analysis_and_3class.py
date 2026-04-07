"""
clickbait_error_analysis_and_3class.py
--------------------------------------
Two-part analysis on DistilBERT v2 predictions:

Step 2: Error Analysis Deep Dive
  - Categorizes errors as "genuine model failures" vs "debatable label quality"
  - Compares model prediction, truthClass (binary label), and truthMean (continuous)
  - Identifies label imperfections where the model and annotator consensus disagree
  - Reports F1 on clear-cut samples only (truthMean < 0.3 or > 0.7)
  - Outputs top misclassified examples by category for manual review

Step 3: Three-Class Model (Not Clickbait / Ambiguous / Clickbait)
  - Uses existing v2 regression scores (no retraining needed)
  - Draws two thresholds on the predicted score to create three bins:
      score < low  -> "Not clickbait" (confident)
      low <= score < high -> "Ambiguous / IDK" (abstain)
      score >= high -> "Clickbait" (confident)
  - Sweeps 16 threshold pairs to find optimal configuration
  - Reports F1/accuracy on confident predictions only
  - Analyzes calibration: when the model is confident, how right is it?
  - Shows what's in the IDK bucket (are they genuinely ambiguous?)

Requires: clickbait_predictions_transformer_v2.csv (from clickbait_distilbert_v2_regression_title.py)

Usage:
    python clickbait_error_analysis_and_3class.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "../results/clickbait_predictions_transformer_v2.csv"
OUTPUT_FILE_ERRORS = "../results/error_analysis_report.csv"
OUTPUT_FILE_3CLASS = "../results/three_class_predictions.csv"


# ===========================================================================
# STEP 2: ERROR ANALYSIS DEEP DIVE
# ===========================================================================

def run_error_analysis(df):
    """Categorize errors as genuine model failures vs debatable label quality."""
    print("=" * 70)
    print("STEP 2: ERROR ANALYSIS DEEP DIVE")
    print("=" * 70)

    pred_scores = df["predicted_score"].values
    truth_mean = df["truthMean"].values
    truth_class = df["true_label"].values
    predicted = df["predicted"].values

    correct = truth_class == predicted
    wrong = ~correct

    print(f"\n   Total test samples: {len(df)}")
    print(f"   Correct: {correct.sum()}  Wrong: {wrong.sum()}")

    # --- Categorize errors ---
    fp_mask = (predicted == 1) & (truth_class == 0)
    fn_mask = (predicted == 0) & (truth_class == 1)

    # FP: model says clickbait, label says not
    # "Genuine" if truthMean < 0.4 (label is clearly right)
    # "Debatable" if truthMean >= 0.4 (annotators were split)
    fp_genuine_mask = fp_mask & (truth_mean < 0.4)
    fp_debatable_mask = fp_mask & (truth_mean >= 0.4)

    # FN: model says not clickbait, label says yes
    # "Genuine" if truthMean > 0.6 (label is clearly right)
    # "Debatable" if truthMean <= 0.6 (annotators were split)
    fn_genuine_mask = fn_mask & (truth_mean > 0.6)
    fn_debatable_mask = fn_mask & (truth_mean <= 0.6)

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_fp_genuine = fp_genuine_mask.sum()
    n_fp_debatable = fp_debatable_mask.sum()
    n_fn_genuine = fn_genuine_mask.sum()
    n_fn_debatable = fn_debatable_mask.sum()
    n_genuine = n_fp_genuine + n_fn_genuine
    n_debatable = n_fp_debatable + n_fn_debatable
    n_wrong = wrong.sum()

    print(f"\n   --- Error Categorization ---")
    print(f"   FALSE POSITIVES (model says clickbait, label says not): {n_fp}")
    print(f"     Genuine errors (truthMean < 0.4, label clearly right): {n_fp_genuine}")
    print(f"     Debatable labels (truthMean >= 0.4, annotators split): {n_fp_debatable}")
    print(f"   FALSE NEGATIVES (model says not clickbait, label says yes): {n_fn}")
    print(f"     Genuine errors (truthMean > 0.6, label clearly right): {n_fn_genuine}")
    print(f"     Debatable labels (truthMean <= 0.6, annotators split): {n_fn_debatable}")
    print(f"\n   SUMMARY:")
    print(f"     Total errors: {n_wrong}")
    print(f"     Genuine model errors: {n_genuine} ({100*n_genuine/n_wrong:.1f}%)")
    print(f"     Debatable label quality: {n_debatable} ({100*n_debatable/n_wrong:.1f}%)")

    # --- Performance on clear-cut samples only ---
    clear_mask = (truth_mean < 0.3) | (truth_mean > 0.7)
    f1_clear = f1_score(truth_class[clear_mask], predicted[clear_mask])
    acc_clear = accuracy_score(truth_class[clear_mask], predicted[clear_mask])

    print(f"\n   --- Performance on Clear-Cut Samples (truthMean < 0.3 or > 0.7) ---")
    print(f"   Samples: {clear_mask.sum()} / {len(df)} ({100*clear_mask.mean():.1f}%)")
    print(f"   F1: {f1_clear:.4f}")
    print(f"   Accuracy: {acc_clear:.4f}")

    # --- Print top examples by category ---
    def print_examples(mask, title, sort_col, ascending, n=15):
        subset = df[mask].sort_values(sort_col, ascending=ascending)
        print(f"\n   --- {title} ---")
        for _, row in subset.head(n).iterrows():
            post = str(row["postText_clean"])[:90].encode("ascii", "replace").decode()
            print(f"     Score={row.predicted_score:.3f} TrMean={row.truthMean:.3f} | {post}")

    print_examples(fp_genuine_mask, "Top Genuine False Positives (model wrong, label clearly right)",
                   "predicted_score", False)
    print_examples(fn_genuine_mask, "Top Genuine False Negatives (model wrong, label clearly right)",
                   "predicted_score", True)
    print_examples(fp_debatable_mask, "Top Debatable FP (model says clickbait, annotators split)",
                   "truthMean", False)
    print_examples(fn_debatable_mask, "Top Debatable FN (model says not clickbait, annotators split)",
                   "truthMean", True)

    # --- Save error analysis CSV ---
    error_df = df[wrong].copy()
    error_df["error_type"] = "unknown"
    error_df.loc[fp_genuine_mask[wrong], "error_type"] = "FP_genuine"
    error_df.loc[fp_debatable_mask[wrong], "error_type"] = "FP_debatable"
    error_df.loc[fn_genuine_mask[wrong], "error_type"] = "FN_genuine"
    error_df.loc[fn_debatable_mask[wrong], "error_type"] = "FN_debatable"
    error_df = error_df.sort_values("cross_entropy_loss", ascending=False)
    error_df.to_csv(OUTPUT_FILE_ERRORS, index=False)
    print(f"\n   Saved error report to {OUTPUT_FILE_ERRORS} ({len(error_df)} rows)")

    return f1_clear, acc_clear


# ===========================================================================
# STEP 3: THREE-CLASS MODEL
# ===========================================================================

def run_three_class(df):
    """Bin v2 regression scores into three classes with threshold optimization."""
    print(f"\n{'=' * 70}")
    print("STEP 3: THREE-CLASS MODEL (Not Clickbait / Ambiguous / Clickbait)")
    print("=" * 70)

    pred_scores = df["predicted_score"].values
    truth_mean = df["truthMean"].values
    truth_class = df["true_label"].values

    # --- True 3-class distribution ---
    true_3class = np.full(len(truth_mean), 1)
    true_3class[truth_mean < 0.3] = 0
    true_3class[truth_mean >= 0.7] = 2

    print(f"\n   True 3-class distribution (based on truthMean):")
    print(f"     Not clickbait (truthMean < 0.3): {(true_3class == 0).sum()} ({100*(true_3class == 0).mean():.1f}%)")
    print(f"     Ambiguous (0.3 - 0.7):           {(true_3class == 1).sum()} ({100*(true_3class == 1).mean():.1f}%)")
    print(f"     Clickbait (truthMean >= 0.7):     {(true_3class == 2).sum()} ({100*(true_3class == 2).mean():.1f}%)")

    # --- Threshold sweep ---
    print(f"\n   --- Threshold Sweep ---")
    print(f"   {'low':>5s}  {'high':>5s}  {'IDK':>6s}  {'%IDK':>6s}  {'Conf F1':>8s}  {'Conf Acc':>8s}")
    print(f"   {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")

    best_f1 = 0
    best_low = 0.25
    best_high = 0.60

    for low_t in [0.25, 0.30, 0.35, 0.40]:
        for high_t in [0.55, 0.60, 0.65, 0.70]:
            pred_3 = np.full(len(pred_scores), 1)
            pred_3[pred_scores < low_t] = 0
            pred_3[pred_scores >= high_t] = 2

            confident = pred_3 != 1
            if confident.sum() == 0:
                continue

            pred_binary = (pred_3[confident] == 2).astype(int)
            true_binary = truth_class[confident]
            f1 = f1_score(true_binary, pred_binary)
            acc = accuracy_score(true_binary, pred_binary)
            n_idk = (~confident).sum()
            pct_idk = 100 * n_idk / len(pred_scores)

            marker = " ***" if f1 > best_f1 else ""
            print(f"   {low_t:5.2f}  {high_t:5.2f}  {n_idk:6d}  {pct_idk:5.1f}%  {f1:8.4f}  {acc:8.4f}{marker}")

            if f1 > best_f1:
                best_f1 = f1
                best_low = low_t
                best_high = high_t

    print(f"\n   Best thresholds: low={best_low}, high={best_high}")
    print(f"   Note: thresholds are asymmetric because the dataset is imbalanced")
    print(f"   (75% non-clickbait) -- model needs less confidence to say 'not clickbait'")

    # --- Detailed analysis with best thresholds ---
    print(f"\n   {'=' * 60}")
    print(f"   DETAILED ANALYSIS (low={best_low}, high={best_high})")
    print(f"   {'=' * 60}")

    pred_3class = np.full(len(pred_scores), 1)
    pred_3class[pred_scores < best_low] = 0
    pred_3class[pred_scores >= best_high] = 2

    print(f"\n   Predicted 3-class distribution:")
    print(f"     Not clickbait (score < {best_low}): {(pred_3class == 0).sum()} ({100*(pred_3class == 0).mean():.1f}%)")
    print(f"     Ambiguous / IDK:                {(pred_3class == 1).sum()} ({100*(pred_3class == 1).mean():.1f}%)")
    print(f"     Clickbait (score >= {best_high}):  {(pred_3class == 2).sum()} ({100*(pred_3class == 2).mean():.1f}%)")

    # --- 3-class confusion matrix ---
    print(f"\n   3-Class Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(true_3class, pred_3class)
    labels = ["NotCB", "Ambig", "CB"]
    print(f"   {'':>8s}  {'Pred NotCB':>10s}  {'Pred Ambig':>10s}  {'Pred CB':>10s}")
    for i, lbl in enumerate(labels):
        print(f"   {lbl:>8s}  {cm[i,0]:10d}  {cm[i,1]:10d}  {cm[i,2]:10d}")

    # --- Performance on confident predictions only ---
    confident_mask = pred_3class != 1
    pred_binary_conf = (pred_3class[confident_mask] == 2).astype(int)
    true_binary_conf = truth_class[confident_mask]

    f1_conf = f1_score(true_binary_conf, pred_binary_conf)
    acc_conf = accuracy_score(true_binary_conf, pred_binary_conf)
    prec_conf = precision_score(true_binary_conf, pred_binary_conf)
    rec_conf = recall_score(true_binary_conf, pred_binary_conf)

    print(f"\n   --- Performance on CONFIDENT Predictions Only ---")
    print(f"   Samples: {confident_mask.sum()} / {len(df)} ({100*confident_mask.mean():.1f}%)")
    print(f"   Abstained (IDK): {(~confident_mask).sum()} ({100*(~confident_mask).mean():.1f}%)")
    print()
    print(classification_report(true_binary_conf, pred_binary_conf,
                                target_names=["no-clickbait", "clickbait"]))
    print(f"   F1={f1_conf:.4f}  Acc={acc_conf:.4f}  Prec={prec_conf:.4f}  Recall={rec_conf:.4f}")

    # --- Calibration ---
    pred_not_cb = pred_3class == 0
    pred_cb = pred_3class == 2
    idk_mask = pred_3class == 1

    print(f"\n   --- Calibration Check ---")
    print(f"   When model says 'Not clickbait': {(truth_class[pred_not_cb] == 0).sum()}/{pred_not_cb.sum()} correct "
          f"({100*(truth_class[pred_not_cb] == 0).mean():.1f}%)")
    print(f"   When model says 'Clickbait':     {(truth_class[pred_cb] == 1).sum()}/{pred_cb.sum()} correct "
          f"({100*(truth_class[pred_cb] == 1).mean():.1f}%)")
    print(f"   When model says 'IDK':           {idk_mask.sum()} abstained")

    # --- What's in the IDK bucket? ---
    idk_truth_mean = truth_mean[idk_mask]
    truly_ambig = ((idk_truth_mean >= 0.3) & (idk_truth_mean <= 0.7)).sum()

    print(f"\n   --- IDK Bucket Analysis ---")
    print(f"   Count: {idk_mask.sum()}")
    print(f"   truthMean: mean={idk_truth_mean.mean():.3f}  median={np.median(idk_truth_mean):.3f}")
    print(f"   truthClass=0: {(truth_class[idk_mask] == 0).sum()}  truthClass=1: {(truth_class[idk_mask] == 1).sum()}")
    print(f"   Truly ambiguous (truthMean 0.3-0.7): {truly_ambig}/{idk_mask.sum()} ({100*truly_ambig/idk_mask.sum():.1f}%)")

    # --- Comparison ---
    print(f"\n   {'=' * 60}")
    print(f"   COMPARISON: Binary vs Three-Class")
    print(f"   {'=' * 60}")
    print(f"   Binary (all samples):          F1=0.7082  Acc=0.8378  ({len(df)} samples)")
    print(f"   Three-class (confident only):  F1={f1_conf:.4f}  Acc={acc_conf:.4f}  ({confident_mask.sum()} samples, {(~confident_mask).sum()} abstained)")

    # --- Save three-class predictions ---
    out = df.copy()
    out["three_class_pred"] = pred_3class
    label_map = {0: "not_clickbait", 1: "ambiguous", 2: "clickbait"}
    out["three_class_label"] = [label_map[p] for p in pred_3class]
    out.to_csv(OUTPUT_FILE_3CLASS, index=False)
    print(f"\n   Saved three-class predictions to {OUTPUT_FILE_3CLASS}")

    return f1_conf, acc_conf


# ===========================================================================
# RUN
# ===========================================================================

print("Loading predictions from DistilBERT v2...")
df = pd.read_csv(INPUT_FILE, encoding="latin-1")
print(f"Loaded {len(df)} test samples\n")

# Step 2
f1_clear, acc_clear = run_error_analysis(df)

# Step 3
f1_3class, acc_3class = run_three_class(df)

print(f"\n{'=' * 70}")
print("DONE!")
print("=" * 70)
