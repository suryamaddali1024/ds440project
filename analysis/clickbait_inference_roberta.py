"""
clickbait_inference_roberta.py
------------------------------
Generalized clickbait classifier using RoBERTa-base (Sanjana's approach).

Mirrors clickbait_inference.py but uses RoBERTa-base instead of DistilBERT.
This lets us compare both architectures head-to-head on the same modern
headlines.

Differences from DistilBERT inference script:
  - Base model: roberta-base (125M params vs DistilBERT's 66M)
  - Input format: postText + " </s> " + targetDescription (single text
    with RoBERTa separator, matching Sanjana's logic)
  - Task: binary classification on truthClass (vs DistilBERT's regression)
  - Class-weighted cross-entropy loss
  - MAX_LENGTH=64 (matching Sanjana's config)
  - Saves to models/roberta_saved/

The script does two things:
  1. TRAIN (first run only): Trains RoBERTa on our dataset and saves
     weights. Takes ~5-6 hours on CPU, ~30 min on GPU.
  2. INFER (subsequent runs): Loads saved weights and classifies any
     new headlines into three categories using clickbait probability.

Input requirements (minimal):
  - A CSV file with at least one text column containing headlines
  - Column name specified via --text_col (default: "headline")

Optional columns (used if present):
  - A second text column for description (--title_col, default: "description")
  - truthClass / label / clickbait columns for evaluation metrics

Usage:
    # First run: trains model + classifies new data
    python clickbait_inference_roberta.py --input headlines.csv --text_col headline

    # Subsequent runs: loads saved model, fast inference only
    python clickbait_inference_roberta.py --input headlines.csv

    # With description for dual input (RoBERTa separator format)
    python clickbait_inference_roberta.py --input data.csv --text_col postText --title_col targetDescription
"""

import argparse
import ast
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===========================================================================
# CONFIG
# ===========================================================================
TRAINING_DATA = "../data/final_cleaned_full.csv"
MODEL_DIR = "../models/roberta_saved"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

# Three-class thresholds applied to clickbait probability (softmax of class 1)
# Default values - may need re-calibration for RoBERTa specifically
THRESHOLD_LOW = 0.30   # below = not clickbait
THRESHOLD_HIGH = 0.70  # above = clickbait

torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ===========================================================================
# HELPERS
# ===========================================================================

def parse_text_list(raw_str):
    """Parse a string-encoded Python list and join elements into plain text."""
    if pd.isna(raw_str) or not str(raw_str).strip():
        return ""
    raw_str = str(raw_str).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            return " ".join([str(item).strip() for item in parsed if str(item).strip()])
        return str(parsed).strip()
    except (ValueError, SyntaxError):
        pass
    if raw_str.startswith("[") and raw_str.endswith("]"):
        raw_str = raw_str[1:-1].strip()
    return raw_str if raw_str else ""


def combine_text(post, desc):
    """
    Combine post + description into a single string with RoBERTa's separator.
    This matches Sanjana's input format.
    """
    if not desc.strip():
        return post
    return post + " </s> " + desc


class ClickbaitTextDataset(Dataset):
    """PyTorch dataset for single text input with binary labels."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
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


class InferenceDataset(Dataset):
    """PyTorch dataset for inference only (no labels)."""

    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


# ===========================================================================
# TRAINING (only runs if saved model not found)
# ===========================================================================

def train_and_save_model():
    """Train RoBERTa on our dataset and save weights."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score

    print("=" * 70)
    print("TRAINING: No saved model found. Training RoBERTa from scratch...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Model: {MODEL_NAME} (Sanjana's approach)")

    # Load training data
    df = pd.read_csv(TRAINING_DATA, encoding="latin-1")
    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)
    df["targetDescription_clean"] = df["targetDescription"].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)

    # Combine post + description with RoBERTa separator (Sanjana's format)
    texts = [
        combine_text(p, d)
        for p, d in zip(df["postText_clean"], df["targetDescription_clean"])
    ]
    labels = df["truthClass"].values

    # Same train/val/test split as our other scripts
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=TEST_SIZE,
        stratify=labels, random_state=RANDOM_STATE,
    )
    train_val_labels = labels[train_val_idx]
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_idx)), test_size=relative_val_size,
        stratify=train_val_labels, random_state=RANDOM_STATE,
    )
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Compute class weights (inverse frequency for imbalanced classes)
    n_neg = np.sum(train_labels == 0)
    n_pos = np.sum(train_labels == 1)
    weight_neg = len(train_labels) / (2 * n_neg)
    weight_pos = len(train_labels) / (2 * n_pos)
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(device)
    print(f"   Class weights: neg={weight_neg:.4f}  pos={weight_pos:.4f}")

    # Tokenize
    print("   Tokenizing...")
    train_dataset = ClickbaitTextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ClickbaitTextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}  Epochs: {EPOCHS}")

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, targets)

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

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        val_f1 = f1_score(val_true, val_preds)
        val_acc = accuracy_score(val_true, val_preds)
        print(f"   Epoch {epoch+1}/{EPOCHS}  Val F1: {val_f1:.4f}  Val Acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"   ** New best **")

    # Restore and save best model
    model.load_state_dict(best_model_state)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n   Model saved to {MODEL_DIR}")

    return model, tokenizer, device


def load_saved_model():
    """Load previously saved RoBERTa weights."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Loading saved model from {MODEL_DIR}")
    print(f"   Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    return model, tokenizer, device


# ===========================================================================
# INFERENCE
# ===========================================================================

def classify_headlines(model, tokenizer, device, texts, descriptions=None):
    """
    Classify headlines using RoBERTa with optional descriptions.

    Args:
        texts: list of headline strings
        descriptions: optional list of description strings (combined with </s>)

    Returns:
        scores (clickbait probabilities), three-class labels
    """
    if descriptions is None:
        combined = texts
    else:
        combined = [combine_text(t, d) for t, d in zip(texts, descriptions)]

    print(f"\n   Classifying {len(combined)} headlines...")
    dataset = InferenceDataset(combined, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Take softmax probability for class 1 (clickbait)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    scores = np.array(all_probs)

    # Apply three-class thresholds
    labels = []
    for s in scores:
        if s < THRESHOLD_LOW:
            labels.append("not_clickbait")
        elif s >= THRESHOLD_HIGH:
            labels.append("clickbait")
        else:
            labels.append("ambiguous")

    return scores, labels


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify headlines as clickbait / ambiguous / not clickbait (RoBERTa)"
    )
    parser.add_argument("--input", required=True, help="Path to CSV with headlines")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--text_col", default="headline", help="Column name for headline text")
    parser.add_argument("--title_col", default=None,
                        help="Optional column for description (RoBERTa separator combine)")
    parser.add_argument("--force_train", action="store_true", help="Force retraining")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_predictions_roberta.csv"

    print("=" * 70)
    print("CLICKBAIT CLASSIFIER (RoBERTa - Sanjana's approach, generalized)")
    print("=" * 70)
    print(f"   Input:    {args.input}")
    print(f"   Output:   {args.output}")
    print(f"   Text col: {args.text_col}")
    print(f"   Desc col: {args.title_col or '(none - single input mode)'}")
    print(f"   Thresholds: < {THRESHOLD_LOW} = not_clickbait, >= {THRESHOLD_HIGH} = clickbait")

    # Load or train model
    if os.path.exists(MODEL_DIR) and not args.force_train:
        print(f"\n   Found saved model at {MODEL_DIR}")
        model, tokenizer, device = load_saved_model()
    else:
        model, tokenizer, device = train_and_save_model()

    # Load input data
    print(f"\n   Loading {args.input}...")
    df = pd.read_csv(args.input, encoding="latin-1")
    print(f"   Loaded {len(df)} rows")

    # Check text column exists
    if args.text_col not in df.columns:
        for alt in ["headline", "title", "text", "postText", "postText_clean", "clean_text"]:
            if alt in df.columns:
                print(f"   Column '{args.text_col}' not found, using '{alt}' instead")
                args.text_col = alt
                break
        else:
            print(f"   ERROR: Column '{args.text_col}' not found.")
            print(f"   Available columns: {list(df.columns)}")
            return

    texts = df[args.text_col].apply(parse_text_list).tolist()

    descriptions = None
    if args.title_col and args.title_col in df.columns:
        descriptions = df[args.title_col].apply(parse_text_list).tolist()
        descriptions = [d if d.strip() else "" for d in descriptions]
        print(f"   Using dual input: {args.text_col} + {args.title_col}")
    else:
        print(f"   Using single input: {args.text_col} only")

    # Classify
    scores, labels = classify_headlines(model, tokenizer, device, texts, descriptions)

    df["predicted_score"] = scores
    df["predicted_label"] = labels

    # Summary
    print(f"\n   --- Classification Results ---")
    for label in ["not_clickbait", "ambiguous", "clickbait"]:
        n = labels.count(label)
        print(f"   {label:15s}: {n:5d} ({100*n/len(labels):.1f}%)")

    # Evaluate if ground truth columns exist
    has_truth = "truthClass" in df.columns or "label" in df.columns or "clickbait" in df.columns
    if has_truth:
        truth_col = next(c for c in ["truthClass", "label", "clickbait"] if c in df.columns)
        true_labels = df[truth_col].values

        from sklearn.metrics import f1_score, accuracy_score, classification_report

        confident_mask = np.array([l != "ambiguous" for l in labels])
        if confident_mask.sum() > 0:
            pred_binary = np.array([1 if l == "clickbait" else 0 for l in labels])[confident_mask]
            true_binary = true_labels[confident_mask]

            f1 = f1_score(true_binary, pred_binary)
            acc = accuracy_score(true_binary, pred_binary)
            print(f"\n   --- Evaluation (ground truth in '{truth_col}') ---")
            print(f"   Confident: {confident_mask.sum()} / {len(labels)} ({100*confident_mask.mean():.1f}%)")
            print(f"   F1: {f1:.4f}  Accuracy: {acc:.4f}")
            print(classification_report(true_binary, pred_binary,
                                        target_names=["no-clickbait", "clickbait"]))
    else:
        print("\n   No ground truth column found -- skipping evaluation metrics")

    df.to_csv(args.output, index=False)
    print(f"   Saved predictions to {args.output}")

    # Sample predictions per class
    print(f"\n   --- Sample Predictions ---")
    for label in ["not_clickbait", "ambiguous", "clickbait"]:
        subset = df[df.predicted_label == label]
        if len(subset) == 0:
            continue
        print(f"\n   {label.upper()} (showing 5):")
        for _, row in subset.sample(min(5, len(subset)), random_state=42).iterrows():
            text = str(row[args.text_col])[:80].encode("ascii", "replace").decode()
            print(f"     Score={row.predicted_score:.3f} | {text}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
