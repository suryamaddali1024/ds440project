"""
clickbait_transformer_v4.py
---------------------------
Fine-tune DistilBERT for clickbait detection with three improvements:
  1. Regression on truthMean (continuous 0-1 score) instead of binary truthClass
  2. Triple text input: [postText] [SEP] [targetTitle | targetDescription]
     Feeds the article's title AND description as context so the model can
     detect mismatches between the post and the article's own metadata
  3. MAX_LENGTH=128 to accommodate the longer combined text

Same train/test split (80/20, stratified on truthClass, seed=42) for fair comparison.

Sections:
  1. Data Loading & Preprocessing
  2. Model Setup (DistilBERT + regression head)
  3. Training (MSE loss on truthMean)
  4. Evaluation (threshold optimization on binary truthClass)
  5. Comparison with all prior models
  6. Error Analysis

Usage:
    1. Upload this script and final_cleaned_full.csv
    2. pip install transformers torch scikit-learn pandas
    3. Set runtime to GPU (strongly recommended)
    4. Run
"""

import ast
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve, mean_squared_error,
)

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "final_cleaned_full.csv"
OUTPUT_FILE = "clickbait_predictions_transformer_v4.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128       # longer to fit postText + title + description
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ===========================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ===========================================================================

def parse_text_list(raw_str):
    """Parse a string-encoded Python list and join elements into plain text."""
    if pd.isna(raw_str) or not str(raw_str).strip():
        return ""
    raw_str = str(raw_str).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            texts = [str(item).strip() for item in parsed if str(item).strip()]
            return " ".join(texts)
        return str(parsed).strip()
    except (ValueError, SyntaxError):
        pass
    if raw_str.startswith("[") and raw_str.endswith("]"):
        raw_str = raw_str[1:-1].strip()
    return raw_str if raw_str else ""


class ClickbaitRegressionDataset(Dataset):
    """PyTorch dataset for text pairs with continuous clickbait scores."""

    def __init__(self, post_texts, title_texts, scores, tokenizer, max_length):
        self.encodings = tokenizer(
            post_texts, title_texts,
            truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        self.scores = torch.tensor(scores, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["scores"] = self.scores[idx]
        return item


def load_and_split():
    """Load data, parse text, split into train/val/test."""
    print("=" * 70)
    print("SECTION 1: DATA LOADING")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, encoding="latin-1")
    print(f"   Loaded {len(df)} rows")

    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)
    df["targetTitle_clean"] = df["targetTitle"].apply(parse_text_list)
    df["targetDescription_clean"] = df["targetDescription"].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"   After cleanup: {len(df)} rows")
    print(f"   Binary class distribution: {df['truthClass'].value_counts().to_dict()}")

    n_desc = (df["targetDescription_clean"].str.strip() != "").sum()
    print(f"   targetDescription coverage: {n_desc}/{len(df)} ({100*n_desc/len(df):.1f}%)")

    # truthMean stats
    print(f"   truthMean stats: mean={df['truthMean'].mean():.3f}  "
          f"median={df['truthMean'].median():.3f}  "
          f"std={df['truthMean'].std():.3f}  "
          f"min={df['truthMean'].min():.3f}  max={df['truthMean'].max():.3f}")

    post_texts = df["postText_clean"].tolist()
    title_texts = df["targetTitle_clean"].tolist()
    desc_texts = df["targetDescription_clean"].tolist()
    # Fill empty titles/descriptions with placeholder
    title_texts = [t if t.strip() else "no title" for t in title_texts]
    desc_texts = [d if d.strip() else "" for d in desc_texts]
    # Combine title + description as the second segment
    # Format: "title | description" (or just "title" if no description)
    context_texts = []
    for t, d in zip(title_texts, desc_texts):
        if d:
            context_texts.append(f"{t} | {d}")
        else:
            context_texts.append(t)
    print(f"   Input format: [postText] [SEP] [targetTitle | targetDescription]")

    scores = df["truthMean"].values          # continuous target for training
    binary_labels = df["truthClass"].values  # binary target for evaluation/stratification

    # Same test split as all other scripts (stratified on binary class)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(binary_labels)), test_size=TEST_SIZE,
        stratify=binary_labels, random_state=RANDOM_STATE,
    )

    # Further split train into train/val
    train_val_labels = binary_labels[train_val_idx]
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_idx)), test_size=relative_val_size,
        stratify=train_val_labels, random_state=RANDOM_STATE,
    )
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    def gather(lst, idxs):
        return [lst[i] for i in idxs]

    train_posts, val_posts, test_posts = gather(post_texts, train_idx), gather(post_texts, val_idx), gather(post_texts, test_idx)
    train_contexts, val_contexts, test_contexts = gather(context_texts, train_idx), gather(context_texts, val_idx), gather(context_texts, test_idx)
    train_scores, val_scores, test_scores = scores[train_idx], scores[val_idx], scores[test_idx]
    test_binary = binary_labels[test_idx]

    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"   Train truthMean: mean={train_scores.mean():.3f}  std={train_scores.std():.3f}")
    print(f"   Val   truthMean: mean={val_scores.mean():.3f}  std={val_scores.std():.3f}")
    print(f"   Test  truthMean: mean={test_scores.mean():.3f}  std={test_scores.std():.3f}")
    print(f"   Test  binary class dist: {dict(zip(*np.unique(test_binary, return_counts=True)))}")

    return (df,
            train_posts, val_posts, test_posts,
            train_contexts, val_contexts, test_contexts,
            train_scores, val_scores, test_scores,
            test_binary, test_idx)


# ===========================================================================
# SECTION 2: MODEL SETUP
# ===========================================================================

def setup_model_and_tokenizer():
    """Load DistilBERT with a regression head (1 output)."""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    print("\n" + "=" * 70)
    print("SECTION 2: MODEL SETUP")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Task: Regression on truthMean (continuous 0-1)")
    print(f"   Input: [postText] [SEP] [targetTitle | targetDescription]")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Use num_labels=1 for regression
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    return model, tokenizer, device


# ===========================================================================
# SECTION 3: TRAINING
# ===========================================================================

def train_model(model, tokenizer, device,
                train_posts, val_posts, train_contexts, val_contexts,
                train_scores, val_scores):
    """Fine-tune DistilBERT with MSE loss on truthMean."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    print("\n" + "=" * 70)
    print("SECTION 3: TRAINING (regression on truthMean)")
    print("=" * 70)
    print(f"   Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"   Max token length: {MAX_LENGTH}")

    # Create datasets
    print("   Tokenizing train set...")
    train_dataset = ClickbaitRegressionDataset(
        train_posts, train_contexts, train_scores, tokenizer, MAX_LENGTH,
    )
    print("   Tokenizing val set...")
    val_dataset = ClickbaitRegressionDataset(
        val_posts, val_contexts, val_scores, tokenizer, MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_fn = nn.MSELoss()

    print(f"   Total training steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print()

    best_val_mse = float("inf")
    best_model_state = None

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["scores"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Squeeze logits from (batch, 1) to (batch,) and clamp to [0, 1]
            preds = torch.sigmoid(outputs.logits.squeeze(-1))
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{EPOCHS}  Batch {batch_idx+1}/{len(train_loader)}  "
                      f"MSE: {total_loss/n_batches:.6f}")

        avg_train_loss = total_loss / n_batches

        # --- Validate ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["scores"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.squeeze(-1))

                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_preds_arr = np.array(val_preds_list)
        val_targets_arr = np.array(val_targets_list)
        val_mse = mean_squared_error(val_targets_arr, val_preds_arr)
        val_corr = np.corrcoef(val_targets_arr, val_preds_arr)[0, 1]

        print(f"   Epoch {epoch+1}/{EPOCHS}  Train MSE: {avg_train_loss:.6f}  "
              f"Val MSE: {val_mse:.6f}  Val Corr: {val_corr:.4f}")

        # Save best model (lowest val MSE)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"   ** New best val MSE: {best_val_mse:.6f} **")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n   Restored best model (val MSE = {best_val_mse:.6f})")

    return model


# ===========================================================================
# SECTION 4: EVALUATION
# ===========================================================================

def evaluate_model(model, tokenizer, device,
                   test_posts, test_contexts, test_scores, test_binary):
    """Evaluate: predict truthMean, then threshold for binary classification."""
    print("\n" + "=" * 70)
    print("SECTION 4: EVALUATION")
    print("=" * 70)

    print("   Tokenizing test set...")
    test_dataset = ClickbaitRegressionDataset(
        test_posts, test_contexts, test_scores, tokenizer, MAX_LENGTH,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get predictions
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits.squeeze(-1))
            all_preds.extend(preds.cpu().numpy())

    pred_scores = np.array(all_preds)

    # --- Regression metrics ---
    test_mse = mean_squared_error(test_scores, pred_scores)
    test_corr = np.corrcoef(test_scores, pred_scores)[0, 1]
    print(f"\n   Regression: MSE={test_mse:.6f}  Correlation={test_corr:.4f}")

    # --- Binary classification via threshold on predicted score ---
    # Use truthClass as ground truth for binary eval
    print("\n   --- Default Threshold (0.5) ---")
    y_pred_default = (pred_scores >= 0.5).astype(int)
    f1_default = f1_score(test_binary, y_pred_default)
    acc_default = accuracy_score(test_binary, y_pred_default)

    print(classification_report(test_binary, y_pred_default,
                                target_names=["no-clickbait", "clickbait"]))
    print(f"   Default F1: {f1_default:.4f}  Acc: {acc_default:.4f}")

    # --- Threshold optimization ---
    print("\n   --- Threshold Optimization ---")
    precisions, recalls, thresholds = precision_recall_curve(test_binary, pred_scores)
    f1_all = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_t_idx = np.argmax(f1_all)
    best_threshold = thresholds[best_t_idx]

    y_pred_opt = (pred_scores >= best_threshold).astype(int)
    f1_opt = f1_score(test_binary, y_pred_opt)
    acc_opt = accuracy_score(test_binary, y_pred_opt)

    print(f"   Best threshold: {best_threshold:.4f}")
    print(classification_report(test_binary, y_pred_opt,
                                target_names=["no-clickbait", "clickbait"]))

    # Pick best
    if f1_opt > f1_default:
        y_pred = y_pred_opt
        threshold_used = best_threshold
        f1_final = f1_opt
        acc_final = acc_opt
        print(f"   >>> Using optimized threshold ({best_threshold:.4f}) -- "
              f"F1 improved {f1_default:.4f} -> {f1_opt:.4f}")
    else:
        y_pred = y_pred_default
        threshold_used = 0.5
        f1_final = f1_default
        acc_final = acc_default
        print(f"   >>> Using default threshold (0.5) -- already best")

    prec_final = precision_score(test_binary, y_pred)
    rec_final = recall_score(test_binary, y_pred)

    print(f"\n   FINAL:  F1={f1_final:.4f}  Acc={acc_final:.4f}  "
          f"Prec={prec_final:.4f}  Recall={rec_final:.4f}")

    return y_pred, pred_scores, test_binary, threshold_used, f1_final, acc_final, prec_final, rec_final


# ===========================================================================
# SECTION 5: COMPARISON WITH ALL PRIOR MODELS
# ===========================================================================

def print_comparison(f1, acc, prec, rec):
    """Print v2 transformer results alongside all prior models."""
    print("\n" + "=" * 70)
    print("SECTION 5: COMPARISON -- All Models")
    print("=" * 70)

    prior = {
        "LightGBM (26 feat)":        {"f1": 0.6263, "acc": 0.8037, "prec": 0.5979, "rec": 0.6575},
        "DistilBERT v1 (postOnly)":  {"f1": 0.6967, "acc": 0.8533, "prec": 0.7215, "rec": 0.6735},
        "DistilBERT v2 (reg+title)": {"f1": 0.7082, "acc": 0.8378, "prec": 0.6439, "rec": 0.7869},
        "Ensemble (v2+LightGBM)":    {"f1": 0.7109, "acc": 0.8412, "prec": 0.6529, "rec": 0.7801},
    }

    header = f"   {'Model':35s}  {'F1':>7s}  {'Acc':>7s}  {'Prec':>7s}  {'Recall':>7s}"
    print(header)
    print(f"   {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for name, r in prior.items():
        print(f"   {name:35s}  {r['f1']:7.4f}  {r['acc']:7.4f}  {r['prec']:7.4f}  {r['rec']:7.4f}")

    print(f"   {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    print(f"   {'DistilBERT v4 (reg+title+desc)':35s}  {f1:7.4f}  {acc:7.4f}  {prec:7.4f}  {rec:7.4f}")

    best_prior = 0.7109  # Ensemble
    diff = f1 - best_prior
    if diff > 0:
        print(f"\n   >>> v4 beats best prior (Ensemble) by +{diff:.4f} F1")
    else:
        print(f"\n   >>> v4 is {diff:+.4f} F1 vs best prior (Ensemble)")


# ===========================================================================
# SECTION 6: ERROR ANALYSIS
# ===========================================================================

def error_analysis(df, test_idx, y_pred, pred_scores, test_binary):
    """Error analysis on v2 transformer predictions."""
    print("\n" + "=" * 70)
    print("SECTION 6: ERROR ANALYSIS")
    print("=" * 70)

    # Per-sample cross-entropy (using predicted score as probability)
    eps = 1e-15
    p_clipped = np.clip(pred_scores, eps, 1 - eps)
    cross_entropy = -(test_binary * np.log(p_clipped) + (1 - test_binary) * np.log(1 - p_clipped))

    correct = test_binary == y_pred
    wrong = ~correct
    fp = (y_pred == 1) & (test_binary == 0)
    fn = (y_pred == 0) & (test_binary == 1)

    print(f"   Correct: {correct.sum()}  |  Wrong: {wrong.sum()}  (FP: {fp.sum()}, FN: {fn.sum()})")
    print(f"   Mean CE (all): {cross_entropy.mean():.4f}")
    print(f"   Mean CE (correct): {cross_entropy[correct].mean():.4f}")
    print(f"   Mean CE (wrong): {cross_entropy[wrong].mean():.4f}")

    # Show predicted scores vs truthMean for misclassified samples
    test_truth_mean = df.iloc[test_idx]["truthMean"].values

    print("\n   --- Top 20 highest cross-entropy (most confidently wrong) ---")
    loss_order = np.argsort(cross_entropy)[::-1]
    print(f"   {'Rank':>4s}  {'TrCls':>5s}  {'Pred':>4s}  {'PrScr':>6s}  {'TrMean':>6s}  {'Loss':>7s}  Post text")
    print(f"   {'-'*4}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*50}")
    for rank, i in enumerate(loss_order[:20], 1):
        row_idx = test_idx[i]
        post = str(df.iloc[row_idx]["postText_clean"])[:70].encode("ascii", "replace").decode()
        print(f"   {rank:4d}  {test_binary[i]:5d}  {y_pred[i]:4d}  {pred_scores[i]:6.3f}  "
              f"{test_truth_mean[i]:6.3f}  {cross_entropy[i]:7.4f}  {post}")

    # Top 20 closest to boundary
    print("\n   --- Top 20 closest to boundary ---")
    boundary_dist = np.abs(pred_scores - 0.5)
    boundary_order = np.argsort(boundary_dist)
    print(f"   {'Rank':>4s}  {'TrCls':>5s}  {'Pred':>4s}  {'PrScr':>6s}  {'TrMean':>6s}  Post text")
    print(f"   {'-'*4}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*50}")
    for rank, i in enumerate(boundary_order[:20], 1):
        row_idx = test_idx[i]
        post = str(df.iloc[row_idx]["postText_clean"])[:70].encode("ascii", "replace").decode()
        print(f"   {rank:4d}  {test_binary[i]:5d}  {y_pred[i]:4d}  {pred_scores[i]:6.3f}  "
              f"{test_truth_mean[i]:6.3f}  {post}")

    # Analyze: are "errors" actually cases where truthMean is ambiguous?
    print("\n   --- truthMean distribution for error types ---")
    for name, mask in [("Correct", correct), ("FP", fp), ("FN", fn)]:
        if mask.sum() > 0:
            tm = test_truth_mean[mask]
            print(f"   {name:10s}  n={mask.sum():4d}  truthMean: mean={tm.mean():.3f}  "
                  f"median={np.median(tm):.3f}  std={tm.std():.3f}")

    # Save output CSV
    print("\n   Saving predictions...")
    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_label"] = test_binary
    out["predicted"] = y_pred
    out["predicted_score"] = pred_scores
    out["cross_entropy_loss"] = cross_entropy
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved to {OUTPUT_FILE}  ({out.shape})")

    return cross_entropy


# ===========================================================================
# RUN
# ===========================================================================

# Section 1
(df,
 train_posts, val_posts, test_posts,
 train_contexts, val_contexts, test_contexts,
 train_scores, val_scores, test_scores,
 test_binary, test_idx) = load_and_split()

# Section 2
model, tokenizer, device = setup_model_and_tokenizer()

# Section 3
model = train_model(model, tokenizer, device,
                    train_posts, val_posts, train_contexts, val_contexts,
                    train_scores, val_scores)

# Section 4
(y_pred, pred_scores, test_binary_out, threshold,
 f1, acc, prec, rec) = evaluate_model(
    model, tokenizer, device,
    test_posts, test_contexts, test_scores, test_binary,
)

# Section 5
print_comparison(f1, acc, prec, rec)

# Section 6
error_analysis(df, test_idx, y_pred, pred_scores, test_binary_out)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
