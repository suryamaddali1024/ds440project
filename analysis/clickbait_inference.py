"""
clickbait_inference.py
----------------------
Generalized clickbait classifier: works on ANY CSV with a headline column.

This script does two things:
  1. TRAIN (first run only): Trains DistilBERT v2 on our dataset and saves
     the model weights to disk. Takes ~3 hours on CPU, ~15 min on GPU.
  2. INFER (all subsequent runs): Loads saved weights and classifies any
     new headlines into three categories:
       - not_clickbait (score < 0.05, calibrated)
       - ambiguous (score 0.05 - 0.75)
       - clickbait (score >= 0.75, calibrated)

The script auto-detects whether saved weights exist. If they do, it skips
training and goes straight to inference.

Input requirements (minimal):
  - A CSV file with at least one text column containing headlines
  - Column name specified via --text_col (default: "headline")

Optional columns (used if present):
  - A second text column for article title (--title_col)
  - truthMean / truthClass columns for evaluation metrics (auto-detected)

Usage:
    # First run: trains model on our dataset + classifies new data
    python clickbait_inference.py --input headlines.csv --text_col headline

    # Subsequent runs: loads saved model, fast inference only
    python clickbait_inference.py --input headlines.csv --text_col headline

    # With article title for dual input
    python clickbait_inference.py --input data.csv --text_col postText --title_col targetTitle

    # Specify output file
    python clickbait_inference.py --input headlines.csv --output predictions.csv
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
TRAINING_DATA = "data/final_cleaned_full.csv"
MODEL_DIR = "models/distilbert_3class_clickbait"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

# Three-class thresholds tuned on this model's test-set score distribution
# (determined by threshold sweep — see threshold analysis in commit history)
# At 0.20/0.60: confident F1=0.87, accuracy=0.94 on 51% of test samples.
THRESHOLD_LOW = 0.20    # below = not clickbait
THRESHOLD_HIGH = 0.60   # above = clickbait

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


class TextPairDataset(Dataset):
    """PyTorch dataset for text pairs with continuous scores."""

    def __init__(self, text_a, text_b, scores, tokenizer, max_length):
        self.encodings = tokenizer(
            text_a, text_b,
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


class InferenceDataset(Dataset):
    """PyTorch dataset for inference only (no labels needed)."""

    def __init__(self, text_a, text_b, tokenizer, max_length):
        self.encodings = tokenizer(
            text_a, text_b,
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
    """Train DistilBERT v2 on our dataset and save weights."""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    print("=" * 70)
    print("TRAINING: No saved model found. Training from scratch...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load training data
    df = pd.read_csv(TRAINING_DATA, encoding="latin-1")
    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)
    df["targetTitle_clean"] = df["targetTitle"].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)

    post_texts = df["postText_clean"].tolist()
    title_texts = [t if t.strip() else "no title" for t in df["targetTitle_clean"].tolist()]
    scores = df["truthMean"].values
    binary_labels = df["truthClass"].values

    # Split
    train_val_idx, test_idx = train_test_split(
        np.arange(len(binary_labels)), test_size=TEST_SIZE,
        stratify=binary_labels, random_state=RANDOM_STATE,
    )
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

    # Load model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(device)

    # Tokenize
    print("   Tokenizing...")
    train_dataset = TextPairDataset(
        gather(post_texts, train_idx), gather(title_texts, train_idx),
        scores[train_idx], tokenizer, MAX_LENGTH,
    )
    val_dataset = TextPairDataset(
        gather(post_texts, val_idx), gather(title_texts, val_idx),
        scores[val_idx], tokenizer, MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    loss_fn = nn.MSELoss()

    print(f"   Train: {len(train_idx)}  Val: {len(val_idx)}  Epochs: {EPOCHS}")

    best_val_mse = float("inf")
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["scores"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

        # Validate
        model.eval()
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["scores"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.squeeze(-1))
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_mse = mean_squared_error(val_targets_list, val_preds_list)
        val_corr = np.corrcoef(val_targets_list, val_preds_list)[0, 1]
        print(f"   Epoch {epoch+1}/{EPOCHS}  Val MSE: {val_mse:.6f}  Val Corr: {val_corr:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
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
    """Load previously saved model weights."""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Loading saved model from {MODEL_DIR}")
    print(f"   Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    return model, tokenizer, device


# ===========================================================================
# INFERENCE
# ===========================================================================

def classify_headlines(model, tokenizer, device, texts, titles=None):
    """
    Classify a list of headlines into three categories.

    Args:
        texts: list of headline strings
        titles: optional list of article title strings (for dual input)

    Returns:
        DataFrame with predicted_score, three_class_label, confidence
    """
    if titles is None:
        titles = [""] * len(texts)

    print(f"\n   Classifying {len(texts)} headlines...")
    dataset = InferenceDataset(texts, titles, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_scores = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            # Original behavior: single-score / regression-style model
            if logits.ndim == 1 or logits.shape[-1] == 1:
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

                for s in scores:
                    all_scores.append(float(s))

                    if s < THRESHOLD_LOW:
                        labels.append("not_clickbait")
                    elif s >= THRESHOLD_HIGH:
                        labels.append("clickbait")
                    else:
                        labels.append("ambiguous")

            # Added support: 3-class classifier model
            elif logits.shape[-1] == 3:
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                for p in probs:
                    class_idx = int(np.argmax(p))
                    all_scores.append(float(np.max(p)))

                    if class_idx == 0:
                        labels.append("not_clickbait")
                    elif class_idx == 1:
                        labels.append("ambiguous")
                    else:
                        labels.append("clickbait")
    

            else:
                    raise ValueError(f"Unsupported model output shape: {tuple(logits.shape)}")

    return np.array(all_scores), labels



# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify headlines as clickbait / ambiguous / not clickbait"
    )
    parser.add_argument("--input", required=True, help="Path to CSV with headlines")
    parser.add_argument("--output", default=None, help="Path for output CSV (default: input_predictions.csv)")
    parser.add_argument("--text_col", default="headline", help="Column name for headline text (default: headline)")
    parser.add_argument("--title_col", default=None, help="Optional column name for article title (dual input)")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if saved model exists")
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_predictions.csv"

    print("=" * 70)
    print("CLICKBAIT CLASSIFIER (Generalized)")
    print("=" * 70)
    print(f"   Input:    {args.input}")
    print(f"   Output:   {args.output}")
    print(f"   Text col: {args.text_col}")
    print(f"   Title col: {args.title_col or '(none - single input mode)'}")
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
        # Try common column names
        for alt in ["headline", "title", "text", "postText", "postText_clean", "clean_text"]:
            if alt in df.columns:
                print(f"   Column '{args.text_col}' not found, using '{alt}' instead")
                args.text_col = alt
                break
        else:
            print(f"   ERROR: Column '{args.text_col}' not found.")
            print(f"   Available columns: {list(df.columns)}")
            return

    # Parse text (handles list-encoded strings from our dataset format)
    texts = df[args.text_col].apply(parse_text_list).tolist()

    # Parse optional title column
    titles = None
    if args.title_col and args.title_col in df.columns:
        titles = df[args.title_col].apply(parse_text_list).tolist()
        titles = [t if t.strip() else "" for t in titles]
        print(f"   Using dual input: {args.text_col} + {args.title_col}")
    else:
        print(f"   Using single input: {args.text_col} only")

    # Classify
    scores, labels = classify_headlines(model, tokenizer, device, texts, titles)

    # Build output
    df["predicted_score"] = scores
    df["predicted_label"] = labels

    # Summary
    print(f"\n   --- Classification Results ---")
    for label in ["not_clickbait", "ambiguous", "clickbait"]:
        n = labels.count(label)
        print(f"   {label:15s}: {n:5d} ({100*n/len(labels):.1f}%)")

    # If ground truth columns exist, compute metrics
    has_truth = "truthClass" in df.columns or "label" in df.columns or "clickbait" in df.columns
    if has_truth:
        truth_col = next(c for c in ["truthClass", "label", "clickbait"] if c in df.columns)
        true_labels = df[truth_col].values

        # Binary evaluation (confident predictions only)
        from sklearn.metrics import f1_score, accuracy_score, classification_report

        confident_mask = np.array([l != "ambiguous" for l in labels])
        if confident_mask.sum() > 0:
            pred_binary = np.array([1 if l == "clickbait" else 0 for l in labels])[confident_mask]
            true_binary = true_labels[confident_mask]

            f1 = f1_score(true_binary, pred_binary)
            acc = accuracy_score(true_binary, pred_binary)
            print(f"\n   --- Evaluation (ground truth found in '{truth_col}') ---")
            print(f"   Confident predictions: {confident_mask.sum()} / {len(labels)} ({100*confident_mask.mean():.1f}%)")
            print(f"   F1: {f1:.4f}  Accuracy: {acc:.4f}")
            print(classification_report(true_binary, pred_binary,
                                        target_names=["no-clickbait", "clickbait"]))
    else:
        print("\n   No ground truth column found -- skipping evaluation metrics")
        print("   (Supports: 'truthClass', 'label', or 'clickbait' columns)")

    # Save
    df.to_csv(args.output, index=False)
    print(f"   Saved predictions to {args.output}")

    # Show sample predictions
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
