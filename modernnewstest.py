from __future__ import annotations

import argparse
import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

CLASS_NAMES = ["not_clickbait", "ambiguous", "clickbait"]
MAX_LENGTH = 96
BATCH_SIZE = 16


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()

    # if stored like "['some text']"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            parts = [str(item).strip() for item in parsed if str(item).strip()]
            return " ".join(parts)
    except Exception:
        pass

    return s


def find_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "postText_clean",
        "postText",
        "targetTitle_clean",
        "targetTitle",
        "headline",
        "title",
        "text",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        return obj_cols[0]

    raise ValueError(
        "Could not find a text column. Expected something like "
        "headline, title, text, postText, or targetTitle."
    )


class HeadlineDataset(Dataset):
    def __init__(self, post_texts, title_texts, tokenizer, max_length):
        self.encodings = tokenizer(
            post_texts,
            title_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Run modern headline test")
    parser.add_argument("--input", required=True, help="Path to modern headlines CSV")
    parser.add_argument("--model_path", default="distilbert-base-uncased", help="Model name or path")
    parser.add_argument("--output", default="modern_headline_predictions.csv", help="Output CSV path")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Tokenizer max length")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Inference batch size")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    

    print("=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    df = pd.read_csv(args.input, encoding="latin-1")
    print(f"Loaded {len(df)} rows")
    print("Columns:", list(df.columns))

    text_col = find_text_column(df)
    print(f"Using text column: {text_col}")

    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.strip() != ""].copy().reset_index(drop=True)
    print(f"Rows after dropping blanks: {len(df)}")

    # match your training input format: [postText] [SEP] [targetTitle]
    df["postText_clean"] = df[text_col]
    df["targetTitle_clean"] = df[text_col]

    print("\n" + "=" * 70)
    print("STEP 2: LOAD MODEL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(device)
    model.eval()

    dataset = HeadlineDataset(
        df["postText_clean"].tolist(),
        df["targetTitle_clean"].tolist(),
        tokenizer,
        args.max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n" + "=" * 70)
    print("STEP 3: RUN PREDICTIONS")
    print("=" * 70)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

    logits = np.vstack(all_logits)
    probs = softmax_np(logits)
    pred_ids = probs.argmax(axis=1)

    df["pred_class_id"] = pred_ids
    df["pred_label"] = [CLASS_NAMES[i] for i in pred_ids]
    df["prob_not_clickbait"] = probs[:, 0]
    df["prob_ambiguous"] = probs[:, 1]
    df["prob_clickbait"] = probs[:, 2]
    df["confidence"] = probs.max(axis=1)

    df.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved predictions to: {args.output}")

    print("\nPrediction distribution:")
    print(df["pred_label"].value_counts(dropna=False).to_string())

    print("\nConfidence summary:")
    print(df["confidence"].describe().to_string())

    print("\nFirst 10 predictions:")
    print(df[[text_col, "pred_label", "confidence"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()