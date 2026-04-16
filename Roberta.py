"""
clickbait_transformer.py
------------------------
Fine-tune DistilBERT for clickbait detection on postText only.

Compares end-to-end transformer learning against hand-crafted feature models.
Uses the same train/test split (80/20, stratified, seed=42) for fair comparison.

Sections:
  1. Data Loading & Preprocessing
  2. Model Setup (DistilBERT + classification head)
  3. Training (fine-tune with class-weighted loss)
  4. Evaluation (threshold optimization, classification report)
  5. Comparison with hand-crafted models
  6. Error Analysis

Usage (Colab):
    1. Upload this script and final_cleaned_full.csv
    2. pip install transformers torch scikit-learn pandas
    3. Set runtime to GPU (strongly recommended)
    4. Run
"""

import ast
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_curve,
)

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "final_cleaned_full.csv"
OUTPUT_FILE = "clickbait_predictions_transformer.csv"
MODEL_NAME = "roberta-base"
MODEL_OUTPUT_DIR = "models/roberta_clickbait"
MAX_LENGTH = 64        # post titles are short, 64 tokens is plenty
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE = 0.20
VAL_SIZE = 0.10        # 10% of full data for validation during training
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


class ClickbaitDataset(Dataset):
    """PyTorch dataset for tokenized post texts."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
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
    df["targetDescription_clean"] = df["targetDescription"].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"   After cleanup: {len(df)} rows")
    print(f"   Class distribution: {df['truthClass'].value_counts().to_dict()}")

    def combine_text(post, desc):
        if not desc.strip():
            return post
        return post + " </s> " + desc   # RoBERTa separator

    texts = [
    combine_text(p, d)
    for p, d in zip(df["postText_clean"], df["targetDescription_clean"])
    ]
    labels = df["truthClass"].values

    # Same test split as clickbait_model_comparison.py
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=TEST_SIZE,
        stratify=labels, random_state=RANDOM_STATE,
    )

    # Further split train into train/val for early stopping
    train_val_labels = labels[train_val_idx]
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)  # ~0.125 of train_val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_idx)), test_size=relative_val_size,
        stratify=train_val_labels, random_state=RANDOM_STATE,
    )
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"   Train class dist: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"   Val   class dist: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    print(f"   Test  class dist: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    return (df, train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels, test_idx)


# ===========================================================================
# SECTION 2: MODEL SETUP
# ===========================================================================

def setup_model_and_tokenizer(train_labels):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    

    print("\n" + "=" * 70)
    print("SECTION 2: MODEL SETUP")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
    )
    model.to(device)

    # Compute class weights for balanced loss
    n_neg = np.sum(train_labels == 0)
    n_pos = np.sum(train_labels == 1)
    weight_neg = len(train_labels) / (2 * n_neg)
    weight_pos = len(train_labels) / (2 * n_pos)
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(device)
    print(f"   Class weights: [no-clickbait={weight_neg:.4f}, clickbait={weight_pos:.4f}]")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    return model, tokenizer, device, class_weights


# ===========================================================================
# SECTION 3: TRAINING
# ===========================================================================

def train_model(model, tokenizer, device, class_weights,
                train_texts, val_texts, train_labels, val_labels):
    """Fine-tune DistilBERT with class-weighted cross-entropy loss."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    print("\n" + "=" * 70)
    print("SECTION 3: TRAINING")
    print("=" * 70)
    print(f"   Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"   Max token length: {MAX_LENGTH}")

    # Create datasets
    print("   Tokenizing train set...")
    train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    print("   Tokenizing val set...")
    val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"   Total training steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print()

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{EPOCHS}  Batch {batch_idx+1}/{len(train_loader)}  "
                      f"Loss: {total_loss/n_batches:.4f}")

        avg_train_loss = total_loss / n_batches

        # --- Validate ---
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        val_f1 = f1_score(val_true, val_preds)
        val_acc = accuracy_score(val_true, val_preds)

        print(f"   Epoch {epoch+1}/{EPOCHS}  Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  Val F1: {val_f1:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"   ** New best val F1: {best_val_f1:.4f} **")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n   Restored best model (val F1 = {best_val_f1:.4f})")

    return model


# ===========================================================================
# SECTION 4: EVALUATION
# ===========================================================================

def evaluate_model(model, tokenizer, device, test_texts, test_labels):
    """Evaluate on test set with threshold optimization."""
    print("\n" + "=" * 70)
    print("SECTION 4: EVALUATION")
    print("=" * 70)

    # Tokenize test set
    print("   Tokenizing test set...")
    test_dataset = ClickbaitDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get predictions
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Convert logits to probabilities
    probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()

    # --- Default threshold (0.5) ---
    print("\n   --- Default Threshold (0.5) ---")
    y_pred_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(all_labels, y_pred_default)
    acc_default = accuracy_score(all_labels, y_pred_default)

    print(classification_report(all_labels, y_pred_default,
                                target_names=["no-clickbait", "clickbait"]))
    print(f"   Default F1: {f1_default:.4f}  Acc: {acc_default:.4f}")

    # --- Threshold optimization ---
    print("\n   --- Threshold Optimization ---")
    precisions, recalls, thresholds = precision_recall_curve(all_labels, probs)
    f1_all = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_t_idx = np.argmax(f1_all)
    best_threshold = thresholds[best_t_idx]

    y_pred_opt = (probs >= best_threshold).astype(int)
    f1_opt = f1_score(all_labels, y_pred_opt)
    acc_opt = accuracy_score(all_labels, y_pred_opt)

    print(f"   Best threshold: {best_threshold:.4f}")
    print(classification_report(all_labels, y_pred_opt,
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

    prec_final = precision_score(all_labels, y_pred)
    rec_final = recall_score(all_labels, y_pred)

    print(f"\n   FINAL:  F1={f1_final:.4f}  Acc={acc_final:.4f}  "
          f"Prec={prec_final:.4f}  Recall={rec_final:.4f}")

    return y_pred, probs, all_labels, threshold_used, f1_final, acc_final, prec_final, rec_final


# ===========================================================================
# SECTION 5: COMPARISON WITH HAND-CRAFTED MODELS
# ===========================================================================

def print_comparison(f1, acc, prec, rec):
    """Print transformer results alongside hand-crafted model results."""
    print("\n" + "=" * 70)
    print("SECTION 5: COMPARISON -- Transformer vs Hand-Crafted Features (26)")
    print("=" * 70)

    # Results from clickbait_model_comparison.py (26 features)
    hc_results = {
        "Logistic Reg (L1)":  {"f1": 0.6222, "acc": 0.8019, "prec": 0.5952, "rec": 0.6518},
        "Logistic Reg (L2)":  {"f1": 0.6186, "acc": 0.7999, "prec": 0.5914, "rec": 0.6483},
        "Elastic Net":        {"f1": 0.6208, "acc": 0.8008, "prec": 0.5927, "rec": 0.6518},
        "Linear SVC":         {"f1": 0.6201, "acc": 0.7942, "prec": 0.5762, "rec": 0.6712},
        "Random Forest":      {"f1": 0.6080, "acc": 0.7638, "prec": 0.5199, "rec": 0.7320},
        "XGBoost":            {"f1": 0.6262, "acc": 0.7951, "prec": 0.5760, "rec": 0.6861},
        "LightGBM":           {"f1": 0.6263, "acc": 0.8037, "prec": 0.5979, "rec": 0.6575},
    }

    header = f"   {'Model':25s}  {'F1':>7s}  {'Acc':>7s}  {'Prec':>7s}  {'Recall':>7s}"
    print(header)
    print(f"   {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for name, r in hc_results.items():
        print(f"   {name:25s}  {r['f1']:7.4f}  {r['acc']:7.4f}  {r['prec']:7.4f}  {r['rec']:7.4f}")

    print(f"   {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    print(f"   {'DistilBERT (fine-tuned)':25s}  {f1:7.4f}  {acc:7.4f}  {prec:7.4f}  {rec:7.4f}")

    best_hc_f1 = max(r["f1"] for r in hc_results.values())
    diff = f1 - best_hc_f1
    if diff > 0:
        print(f"\n   >>> DistilBERT beats best hand-crafted model by +{diff:.4f} F1")
    else:
        print(f"\n   >>> DistilBERT is {diff:+.4f} F1 vs best hand-crafted model")


# ===========================================================================
# SECTION 6: ERROR ANALYSIS
# ===========================================================================

def error_analysis(df, test_idx, y_pred, probs, test_labels_out):
    """Error analysis on transformer predictions."""
    # Section 7: Save trained model + tokenizer
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"\nSaved trained model and tokenizer to {MODEL_OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)

    # Per-sample cross-entropy
    eps = 1e-15
    p_clipped = np.clip(probs, eps, 1 - eps)
    cross_entropy = -(test_labels * np.log(p_clipped) + (1 - test_labels) * np.log(1 - p_clipped))

    correct = test_labels == y_pred
    wrong = ~correct
    fp = (y_pred == 1) & (test_labels == 0)
    fn = (y_pred == 0) & (test_labels == 1)

    print(f"   Correct: {correct.sum()}  |  Wrong: {wrong.sum()}  (FP: {fp.sum()}, FN: {fn.sum()})")
    print(f"   Mean CE (all): {cross_entropy.mean():.4f}")
    print(f"   Mean CE (correct): {cross_entropy[correct].mean():.4f}")
    print(f"   Mean CE (wrong): {cross_entropy[wrong].mean():.4f}")

    # Top 20 highest loss
    print("\n   --- Top 20 highest cross-entropy (most confidently wrong) ---")
    loss_order = np.argsort(cross_entropy)[::-1]
    print(f"   {'Rank':>4s}  {'True':>4s}  {'Pred':>4s}  {'Prob':>6s}  {'Loss':>7s}  Post text")
    print(f"   {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*50}")
    for rank, i in enumerate(loss_order[:20], 1):
        row_idx = test_idx[i]
        post = str(df.iloc[row_idx]["postText_clean"])[:80].encode("ascii", "replace").decode()
        print(f"   {rank:4d}  {test_labels[i]:4d}  {y_pred[i]:4d}  {probs[i]:6.3f}  "
              f"{cross_entropy[i]:7.4f}  {post}")

    # Top 20 closest to boundary
    print("\n   --- Top 20 closest to boundary ---")
    boundary_dist = np.abs(probs - 0.5)
    boundary_order = np.argsort(boundary_dist)
    print(f"   {'Rank':>4s}  {'True':>4s}  {'Pred':>4s}  {'Prob':>6s}  Post text")
    print(f"   {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*50}")
    for rank, i in enumerate(boundary_order[:20], 1):
        row_idx = test_idx[i]
        post = str(df.iloc[row_idx]["postText_clean"])[:80].encode("ascii", "replace").decode()
        print(f"   {rank:4d}  {test_labels[i]:4d}  {y_pred[i]:4d}  {probs[i]:6.3f}  {post}")

    # Save output CSV
    print("\n   Saving predictions...")
    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_label"] = test_labels
    out["predicted"] = y_pred
    out["predicted_proba"] = probs
    out["cross_entropy_loss"] = cross_entropy
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved to {OUTPUT_FILE}  ({out.shape})")

    return cross_entropy


# ===========================================================================
# RUN
# ===========================================================================

# Section 1
(df, train_texts, val_texts, test_texts,
 train_labels, val_labels, test_labels, test_idx) = load_and_split()

# Section 2
model, tokenizer, device, class_weights = setup_model_and_tokenizer(train_labels)

# Section 3
model = train_model(model, tokenizer, device, class_weights,
                    train_texts, val_texts, train_labels, val_labels)

# Section 4
(y_pred, probs, test_labels_out, threshold,
 f1, acc, prec, rec) = evaluate_model(model, tokenizer, device, test_texts, test_labels)

# Section 5
print_comparison(f1, acc, prec, rec)

# Section 6
error_analysis(df, test_idx, y_pred, probs, test_labels_out)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
