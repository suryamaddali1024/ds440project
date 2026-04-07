"""
generate_bart_summaries.py
--------------------------
Generate factual baseline summaries of each article's targetParagraphs using
facebook/bart-large-cnn.  Summaries are length-matched to the corresponding
postText so downstream SBERT cosine-similarity comparisons are meaningful.

Usage (Colab):
    1. Upload this script and final_cleaned_full.csv to Colab
    2. pip install transformers  (torch is pre-installed)
    3. Set runtime to GPU
    4. Paste this entire script into a cell and run
"""

import ast
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===========================================================================
# CONFIG — edit these as needed
# ===========================================================================
INPUT_FILE = "../data/final_cleaned_full.csv"
OUTPUT_FILE = "../data/final_cleaned_with_summaries.csv"
CHECKPOINT_FILE = "../data/bart_checkpoint.csv"
SAVE_EVERY = 500          # checkpoint interval (rows)
RESUME = False            # set to True to resume from checkpoint
DEVICE = "auto"           # "cuda", "cpu", or "auto"

# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def trim_to_complete_sentences(text):
    """Trim text to the last complete sentence (ending with . ! or ?).

    If no sentence boundary is found, return the original text as-is.
    """
    if not text or not text.strip():
        return ""
    # Match a sentence-ending period only when preceded by a word of 2+ chars
    # (avoids abbreviations like "U.S." or "Dr."), OR match ! / ?.
    matches = list(re.finditer(
        r'(?:[a-z]{2,}|[0-9])[.]["\')]*(?:\s|$)|[!?]["\')]*(?:\s|$)',
        text, re.IGNORECASE
    ))
    if matches:
        last = matches[-1]
        return text[:last.end()].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Summary length parameters
# ---------------------------------------------------------------------------

def compute_summary_params(posttext_word_count):
    """Convert postText word count to BART min_length / max_length token params.

    Heuristic: ~1.33 BART tokens per English word.
    min_length = 60% of target (floor 5)
    max_length = 200% of target (floor 40, cap 100)

    The floor of 40 tokens ensures BART always has enough room to produce
    complete sentences, even for short postTexts (~12 words).
    """
    if posttext_word_count <= 0:
        return 5, 45

    target_tokens = max(1, round(posttext_word_count * 1.33))
    min_length = max(5, math.floor(target_tokens * 0.6))
    max_length = max(40, min(100, math.ceil(target_tokens * 2.0)))

    if min_length >= max_length:
        max_length = min_length + 5

    return min_length, max_length


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_data(input_path):
    """Load CSV, parse text columns, compute word counts."""
    print(f"1. Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    print("2. Parsing postText and targetParagraphs...")
    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)

    df["postText_word_count"] = df["postText_clean"].apply(
        lambda x: len(x.split()) if x else 0
    )

    empty_articles = (df["targetParagraphs_clean"] == "").sum()
    empty_posts = (df["postText_clean"] == "").sum()
    print(f"   Empty articles: {empty_articles}")
    print(f"   Empty posts: {empty_posts}")
    print(f"   Median postText word count: {df['postText_word_count'].median():.0f}")

    return df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_bart_model(device_arg):
    """Load facebook/bart-large-cnn tokenizer + model."""
    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    print(f"3. Loading BART model on {device}...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model = model.to(device)
    model.eval()
    print("   Model loaded successfully")

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summaries(df, tokenizer, model, device):
    """Generate summaries row-by-row, with checkpoint support."""
    import torch

    # Determine starting index for resume
    if RESUME and "generatedSummary" in df.columns:
        start_idx = (df["generatedSummary"].fillna("").astype(str).str.strip() != "").sum()
        print(f"4. Resuming from row {start_idx}...")
    else:
        df["generatedSummary"] = pd.array([""] * len(df), dtype="string")
        start_idx = 0
        print("4. Starting summary generation...")

    end_idx = len(df)
    total_to_process = end_idx - start_idx
    if total_to_process <= 0:
        print("   Nothing to process.")
        return df

    print(f"   Processing rows {start_idx} to {end_idx - 1} ({total_to_process} rows)")

    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(start_idx, end_idx), desc="Generating summaries",
                      initial=start_idx, total=end_idx):
            article = df.at[i, "targetParagraphs_clean"]
            posttext_wc = df.at[i, "postText_word_count"]

            # Skip empty articles
            if not article or not article.strip():
                df.at[i, "generatedSummary"] = ""
                continue

            # Compute dynamic length params
            min_len, max_len = compute_summary_params(posttext_wc)

            # Tokenize (truncate long articles to 1024 tokens)
            inputs = tokenizer(
                article,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_length=min_len,
                max_length=max_len,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary_text = trim_to_complete_sentences(summary_text)
            df.at[i, "generatedSummary"] = summary_text

            # Checkpoint
            if SAVE_EVERY and (i + 1) % SAVE_EVERY == 0:
                df.to_csv(CHECKPOINT_FILE, index=False)
                elapsed = time.time() - start_time
                rows_done = i + 1 - start_idx
                rate = elapsed / rows_done if rows_done else 0
                remaining = rate * (end_idx - i - 1)
                tqdm.write(
                    f"   Checkpoint saved at row {i + 1} "
                    f"({rate:.1f}s/row, ~{remaining/3600:.1f}h remaining)"
                )

    elapsed = time.time() - start_time
    print(f"   Done! {total_to_process} rows in {elapsed:.1f}s "
          f"({elapsed/total_to_process:.2f}s/row)")

    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

INTERMEDIATE_COLS = ["postText_clean", "targetParagraphs_clean", "postText_word_count"]


def save_final_output(df, output_path):
    """Drop intermediate columns and save final CSV."""
    out = df.drop(columns=[c for c in INTERMEDIATE_COLS if c in df.columns])
    out.to_csv(output_path, index=False)
    print(f"5. Saved to {output_path}")
    print(f"   Shape: {out.shape}")
    print(f"   Columns: {out.columns.tolist()}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def print_verification_stats(df):
    """Print summary statistics for the full run."""
    print("\n" + "=" * 70)
    print("VERIFICATION STATISTICS")
    print("=" * 70)

    has_summary = df["generatedSummary"].notna() & (df["generatedSummary"] != "")
    sdf = df[has_summary].copy()

    if len(sdf) == 0:
        print("  No summaries generated.")
        return

    sdf["summary_word_count"] = sdf["generatedSummary"].apply(lambda x: len(str(x).split()))

    print(f"\n  Total rows with summaries: {len(sdf)}")
    print(f"  Rows with empty summaries: {(~has_summary).sum()}")

    print(f"\n  postText word count distribution:")
    print(f"    {sdf['postText_word_count'].describe().to_string()}")

    print(f"\n  Summary word count distribution:")
    print(f"    {sdf['summary_word_count'].describe().to_string()}")

    sdf["length_ratio"] = sdf["summary_word_count"] / sdf["postText_word_count"].clip(lower=1)
    print(f"\n  Summary/postText length ratio:")
    print(f"    {sdf['length_ratio'].describe().to_string()}")

    good_ratio = ((sdf["length_ratio"] >= 0.5) & (sdf["length_ratio"] <= 2.0)).mean()
    print(f"\n  Rows with ratio in [0.5, 2.0]: {good_ratio:.1%}")

    print(f"\n  5 random sample comparisons:")
    sample_idx = random.sample(range(len(sdf)), min(5, len(sdf)))
    for idx in sample_idx:
        row = sdf.iloc[idx]
        post = row["postText_clean"] if "postText_clean" in row.index else "(unavailable)"
        summary = row["generatedSummary"]
        post_wc = len(post.split()) if post else 0
        summary_wc = len(str(summary).split())
        print(f"\n    Row {row.name}:")
        print(f"      postText ({post_wc}w): {post[:100]}")
        print(f"      summary  ({summary_wc}w): {str(summary)[:100]}")

    print("\n" + "=" * 70)


# ===========================================================================
# RUN
# ===========================================================================

# Load data
if RESUME and os.path.exists(CHECKPOINT_FILE):
    print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
    df = load_and_preprocess_data(CHECKPOINT_FILE)
else:
    df = load_and_preprocess_data(INPUT_FILE)

# Load model
tokenizer, model, device = load_bart_model(DEVICE)

# Generate summaries
df = generate_summaries(df, tokenizer, model, device)

# Save final output
save_final_output(df, OUTPUT_FILE)

# Clean up checkpoint on successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"   Removed checkpoint file: {CHECKPOINT_FILE}")

# Print stats
print_verification_stats(df)

print("\nDone!")
