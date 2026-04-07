"""
clickbait_distilbert_v5_3class.py
---------------------------------
Fine-tune DistilBERT as a three-class classifier:
  Class 0: Not clickbait (truthMean < 0.3)
  Class 1: Ambiguous (truthMean 0.3 - 0.7)
  Class 2: Clickbait (truthMean >= 0.7)

The model explicitly learns to say "I don't know" for borderline cases
rather than being forced to pick a side. This addresses the finding that
47% of binary model errors are debatable labels, not genuine failures.

Input: [postText] [SEP] [targetTitle] (same as v2)

Same test split (80/20, stratified on truthClass, seed=42) for fair comparison.

Sections:
  1. Data Loading & Preprocessing (truthMean -> 3-class labels)
  2. Model Setup (DistilBERT + 3-class head with class weights)
  3. Training (class-weighted cross-entropy)
  4. Evaluation (3-class metrics + confident-only binary metrics)
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
    classification_report, confusion_matrix,
)

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "../data/final_cleaned_full.csv"
OUTPUT_FILE = "../results/clickbait_predictions_3class.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

# Three-class thresholds on truthMean
AMBIG_LOW = 0.3    # below this = not clickbait
AMBIG_HIGH = 0.7   # above this = clickbait

CLASS_NAMES = ["not_clickbait", "ambiguous", "clickbait"]

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


def truthmean_to_3class(truth_mean):
    """Convert continuous truthMean to three-class labels."""
    labels = np.full(len(truth_mean), 1)    # default: ambiguous
    labels[truth_mean < AMBIG_LOW] = 0      # not clickbait
    labels[truth_mean >= AMBIG_HIGH] = 2    # clickbait
    return labels


class ClickbaitThreeClassDataset(Dataset):
    """PyTorch dataset for text pairs with three-class labels."""

    def __init__(self, post_texts, title_texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            post_texts, title_texts,
            truncation=True, padding=True,
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
    """Load data, parse text, create 3-class labels, split into train/val/test."""
    print("=" * 70)
    print("SECTION 1: DATA LOADING")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, encoding="latin-1")
    print(f"   Loaded {len(df)} rows")

    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)
    df["targetTitle_clean"] = df["targetTitle"].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"   After cleanup: {len(df)} rows")

    post_texts = df["postText_clean"].tolist()
    title_texts = df["targetTitle_clean"].tolist()
    title_texts = [t if t.strip() else "no title" for t in title_texts]

    truth_mean = df["truthMean"].values
    binary_labels = df["truthClass"].values
    three_class_labels = truthmean_to_3class(truth_mean)

    print(f"   Binary class distribution: {df['truthClass'].value_counts().to_dict()}")
    print(f"   Three-class distribution (truthMean thresholds {AMBIG_LOW}/{AMBIG_HIGH}):")
    for i, name in enumerate(CLASS_NAMES):
        n = (three_class_labels == i).sum()
        print(f"     {name}: {n} ({100*n/len(three_class_labels):.1f}%)")

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

    train_posts = gather(post_texts, train_idx)
    val_posts = gather(post_texts, val_idx)
    test_posts = gather(post_texts, test_idx)
    train_titles = gather(title_texts, train_idx)
    val_titles = gather(title_texts, val_idx)
    test_titles = gather(title_texts, test_idx)

    train_labels = three_class_labels[train_idx]
    val_labels = three_class_labels[val_idx]
    test_labels = three_class_labels[test_idx]
    test_binary = binary_labels[test_idx]

    print(f"\n   Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        dist = {CLASS_NAMES[i]: (split_labels == i).sum() for i in range(3)}
        print(f"   {split_name} 3-class dist: {dist}")

    return (df,
            train_posts, val_posts, test_posts,
            train_titles, val_titles, test_titles,
            train_labels, val_labels, test_labels,
            test_binary, test_idx)


# ===========================================================================
# SECTION 2: MODEL SETUP
# ===========================================================================

def setup_model_and_tokenizer(train_labels):
    """Load DistilBERT with 3-class head and compute class weights."""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    print("\n" + "=" * 70)
    print("SECTION 2: MODEL SETUP")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Task: 3-class classification (not_clickbait / ambiguous / clickbait)")
    print(f"   Input: [postText] [SEP] [targetTitle]")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
    )
    model.to(device)

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = len(train_labels) / (3 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"   Class counts: {dict(zip(CLASS_NAMES, class_counts))}")
    print(f"   Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")

    return model, tokenizer, device, class_weights_tensor


# ===========================================================================
# SECTION 3: TRAINING
# ===========================================================================

def train_model(model, tokenizer, device, class_weights,
                train_posts, val_posts, train_titles, val_titles,
                train_labels, val_labels):
    """Fine-tune DistilBERT with class-weighted cross-entropy for 3 classes."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    print("\n" + "=" * 70)
    print("SECTION 3: TRAINING (3-class classification)")
    print("=" * 70)
    print(f"   Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"   Max token length: {MAX_LENGTH}")

    print("   Tokenizing train set...")
    train_dataset = ClickbaitThreeClassDataset(
        train_posts, train_titles, train_labels, tokenizer, MAX_LENGTH,
    )
    print("   Tokenizing val set...")
    val_dataset = ClickbaitThreeClassDataset(
        val_posts, val_titles, val_labels, tokenizer, MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_f1_macro = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)

        print(f"   Epoch {epoch+1}/{EPOCHS}  Train Loss: {avg_train_loss:.4f}  "
              f"Val F1(macro): {val_f1_macro:.4f}  Val Acc: {val_acc:.4f}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"   ** New best val F1(macro): {best_val_f1:.4f} **")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n   Restored best model (val F1 macro = {best_val_f1:.4f})")

    return model


# ===========================================================================
# SECTION 4: EVALUATION
# ===========================================================================

def evaluate_model(model, tokenizer, device,
                   test_posts, test_titles, test_labels, test_binary):
    """Evaluate 3-class predictions + confident-only binary metrics."""
    print("\n" + "=" * 70)
    print("SECTION 4: EVALUATION")
    print("=" * 70)

    print("   Tokenizing test set...")
    test_dataset = ClickbaitThreeClassDataset(
        test_posts, test_titles, test_labels, tokenizer, MAX_LENGTH,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    pred_3class = np.array(all_preds)
    pred_probs = np.concatenate(all_probs, axis=0)  # shape: (n, 3)

    # --- 3-class metrics ---
    print("\n   --- 3-Class Classification Report ---")
    print(classification_report(test_labels, pred_3class, target_names=CLASS_NAMES))

    f1_macro = f1_score(test_labels, pred_3class, average="macro")
    acc_3class = accuracy_score(test_labels, pred_3class)
    print(f"   3-class F1(macro): {f1_macro:.4f}  Accuracy: {acc_3class:.4f}")

    print("\n   3-Class Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(test_labels, pred_3class)
    print(f"   {'':>15s}  {'Pred NotCB':>10s}  {'Pred Ambig':>10s}  {'Pred CB':>10s}")
    for i, lbl in enumerate(CLASS_NAMES):
        print(f"   {lbl:>15s}  {cm[i,0]:10d}  {cm[i,1]:10d}  {cm[i,2]:10d}")

    # --- Confident-only binary metrics ---
    # Exclude samples the model predicts as "ambiguous" (class 1)
    confident_mask = pred_3class != 1

    print(f"\n   --- Confident Predictions Only (excluding 'ambiguous') ---")
    print(f"   Confident: {confident_mask.sum()} / {len(pred_3class)} ({100*confident_mask.mean():.1f}%)")
    print(f"   Abstained (ambiguous): {(~confident_mask).sum()} ({100*(~confident_mask).mean():.1f}%)")

    if confident_mask.sum() > 0:
        # Map 3-class to binary: class 0 -> 0, class 2 -> 1
        pred_binary_conf = (pred_3class[confident_mask] == 2).astype(int)
        true_binary_conf = test_binary[confident_mask]

        f1_conf = f1_score(true_binary_conf, pred_binary_conf)
        acc_conf = accuracy_score(true_binary_conf, pred_binary_conf)
        prec_conf = precision_score(true_binary_conf, pred_binary_conf)
        rec_conf = recall_score(true_binary_conf, pred_binary_conf)

        print()
        print(classification_report(true_binary_conf, pred_binary_conf,
                                    target_names=["no-clickbait", "clickbait"]))
        print(f"   Confident F1={f1_conf:.4f}  Acc={acc_conf:.4f}  "
              f"Prec={prec_conf:.4f}  Recall={rec_conf:.4f}")

        # Calibration
        pred_not_cb = pred_3class == 0
        pred_cb = pred_3class == 2
        print(f"\n   --- Calibration ---")
        print(f"   Says 'not clickbait': {(test_binary[pred_not_cb] == 0).sum()}/{pred_not_cb.sum()} correct "
              f"({100*(test_binary[pred_not_cb] == 0).mean():.1f}%)")
        print(f"   Says 'clickbait':     {(test_binary[pred_cb] == 1).sum()}/{pred_cb.sum()} correct "
              f"({100*(test_binary[pred_cb] == 1).mean():.1f}%)")
    else:
        f1_conf = 0.0
        acc_conf = 0.0
        prec_conf = 0.0
        rec_conf = 0.0

    return pred_3class, pred_probs, f1_macro, acc_3class, f1_conf, acc_conf, prec_conf, rec_conf


# ===========================================================================
# SECTION 5: COMPARISON
# ===========================================================================

def print_comparison(f1_macro, acc_3class, f1_conf, acc_conf):
    """Print results alongside all prior models."""
    print("\n" + "=" * 70)
    print("SECTION 5: COMPARISON")
    print("=" * 70)

    print("\n   --- Binary models (forced prediction on all samples) ---")
    print(f"   {'Model':35s}  {'F1':>7s}  {'Acc':>7s}")
    print(f"   {'-'*35}  {'-'*7}  {'-'*7}")
    print(f"   {'LightGBM (26 feat)':35s}  {'0.6263':>7s}  {'0.8037':>7s}")
    print(f"   {'DistilBERT v2 (reg+title)':35s}  {'0.7082':>7s}  {'0.8378':>7s}")
    print(f"   {'Ensemble (v2+LightGBM)':35s}  {'0.7109':>7s}  {'0.8412':>7s}")

    print(f"\n   --- Three-class models (can abstain on ambiguous) ---")
    print(f"   {'Model':35s}  {'Conf F1':>7s}  {'Conf Acc':>8s}  {'%Conf':>6s}")
    print(f"   {'-'*35}  {'-'*7}  {'-'*8}  {'-'*6}")
    print(f"   {'v2 binned (post-hoc thresholds)':35s}  {'0.8376':>7s}  {'0.9369':>8s}  {'63.6%':>6s}")
    print(f"   {'v5 trained 3-class (this model)':35s}  {f1_conf:7.4f}  {acc_conf:8.4f}  {'TBD':>6s}")

    print(f"\n   v5 3-class macro F1: {f1_macro:.4f}  3-class Acc: {acc_3class:.4f}")


# ===========================================================================
# SECTION 6: ERROR ANALYSIS
# ===========================================================================

def error_analysis(df, test_idx, pred_3class, pred_probs, test_labels, test_binary):
    """Analyze what the 3-class model gets right and wrong."""
    print("\n" + "=" * 70)
    print("SECTION 6: ERROR ANALYSIS")
    print("=" * 70)

    truth_mean = df.iloc[test_idx]["truthMean"].values

    # What's in each predicted bucket?
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = pred_3class == cls_id
        if mask.sum() == 0:
            continue
        tm = truth_mean[mask]
        binary = test_binary[mask]
        print(f"\n   Predicted '{cls_name}' ({mask.sum()} samples):")
        print(f"     truthMean: mean={tm.mean():.3f}  median={np.median(tm):.3f}  std={tm.std():.3f}")
        print(f"     truthClass=0: {(binary == 0).sum()}  truthClass=1: {(binary == 1).sum()}")

    # Confidence analysis
    max_probs = pred_probs.max(axis=1)
    correct_3class = pred_3class == test_labels
    print(f"\n   --- Model Confidence ---")
    print(f"   Mean confidence (all):     {max_probs.mean():.4f}")
    print(f"   Mean confidence (correct): {max_probs[correct_3class].mean():.4f}")
    print(f"   Mean confidence (wrong):   {max_probs[~correct_3class].mean():.4f}")

    # Top misclassified in 3-class
    print(f"\n   --- Top 15 3-class misclassifications ---")
    wrong_mask = ~correct_3class
    wrong_idx = np.where(wrong_mask)[0]
    wrong_conf = max_probs[wrong_mask]
    wrong_order = np.argsort(wrong_conf)[::-1]  # most confident first

    print(f"   {'True':>8s}  {'Pred':>8s}  {'Conf':>6s}  {'TrMean':>6s}  Post text")
    print(f"   {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*50}")
    for rank, wi in enumerate(wrong_order[:15]):
        i = wrong_idx[wi]
        row_idx = test_idx[i]
        post = str(df.iloc[row_idx]["postText_clean"])[:70].encode("ascii", "replace").decode()
        print(f"   {CLASS_NAMES[test_labels[i]]:>8s}  {CLASS_NAMES[pred_3class[i]]:>8s}  "
              f"{max_probs[i]:6.3f}  {truth_mean[i]:6.3f}  {post}")

    # Save output
    print("\n   Saving predictions...")
    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_3class"] = test_labels
    out["pred_3class"] = pred_3class
    out["true_3class_label"] = [CLASS_NAMES[l] for l in test_labels]
    out["pred_3class_label"] = [CLASS_NAMES[p] for p in pred_3class]
    out["prob_not_clickbait"] = pred_probs[:, 0]
    out["prob_ambiguous"] = pred_probs[:, 1]
    out["prob_clickbait"] = pred_probs[:, 2]
    out["true_binary"] = test_binary
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved to {OUTPUT_FILE} ({out.shape})")


# ===========================================================================
# RUN
# ===========================================================================

# Section 1
(df,
 train_posts, val_posts, test_posts,
 train_titles, val_titles, test_titles,
 train_labels, val_labels, test_labels,
 test_binary, test_idx) = load_and_split()

# Section 2
model, tokenizer, device, class_weights = setup_model_and_tokenizer(train_labels)

# Section 3
model = train_model(model, tokenizer, device, class_weights,
                    train_posts, val_posts, train_titles, val_titles,
                    train_labels, val_labels)

# Section 4
(pred_3class, pred_probs, f1_macro, acc_3class,
 f1_conf, acc_conf, prec_conf, rec_conf) = evaluate_model(
    model, tokenizer, device,
    test_posts, test_titles, test_labels, test_binary,
)

# Section 5
print_comparison(f1_macro, acc_3class, f1_conf, acc_conf)

# Section 6
error_analysis(df, test_idx, pred_3class, pred_probs, test_labels, test_binary)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
