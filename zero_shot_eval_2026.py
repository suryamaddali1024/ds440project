"""
zero_shot_eval_2026.py
-----------------------
Zero-Shot Evaluation of the fine-tuned DistilBERT 3-class clickbait model
on 2026 news headlines.

What "zero-shot" means here
----------------------------
The model was trained exclusively on the Webis-Clickbait-17 corpus
(Facebook posts + article titles from 2016-2017).  It has never seen any
2026 headlines.  Running it on modern news tests whether the linguistic
patterns it learnt are *general* enough to survive temporal drift, topic
drift, and the uniquely sensational language of April Fools 2026.

Classes
-------
  0  Not Clickbait   (truthMean < 0.30 during training)
  1  Ambiguous       (truthMean 0.30 – 0.70)
  2  Clickbait       (truthMean >= 0.70)

Outputs
-------
  • Console: rich comparison table with confidence bars
  • Console: April Fools deep-dive block
  • Console: per-category summary statistics
  • File:    zero_shot_eval_results_2026.csv

Usage
-----
    python zero_shot_eval_2026.py                          # uses random weights (demo)
    python zero_shot_eval_2026.py --weights distilbert_v5_3class.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 96
NUM_LABELS  = 3
OUTPUT_FILE = "zero_shot_eval_results_2026.csv"

LABEL_MAP   = {0: "Not Clickbait", 1: "Ambiguous", 2: "Clickbait"}
SHORT_MAP   = {0: "Not CB", 1: "Ambig", 2: "Clickbait"}
EMOJI_MAP   = {0: "✓", 1: "~", 2: "✗"}

# Confidence bar settings
BAR_WIDTH   = 20       # max characters in the intensity bar
BAR_CHAR    = "█"
EMPTY_CHAR  = "░"

# April Fools detection: date string embedded in category names
APRIL_FOOLS_TAG = "april_fools_2026"


# ===========================================================================
# SECTION 1 — HEADLINES DATASET
# ===========================================================================

# Each entry: (category, headline)
# Categories:
#   "factual_2026"      – straightforward, verifiable 2026 news
#   "soft_clickbait"    – modern soft-sensationalism; may land on Ambiguous
#   "hard_clickbait"    – classic clickbait language, model should flag class 2
#   "april_fools_2026"  – published on 1 April 2026; sensational but potentially true
#   "miami_herald"      – real-framing edge cases for error analysis (Task 1 analogue)

RAW_HEADLINES: List[Tuple[str, str]] = [

    # ── Factual / Not Clickbait ─────────────────────────────────────────────
    ("factual_2026", "Federal Reserve holds interest rates steady at 4.25% for third consecutive meeting"),
    ("factual_2026", "Senate confirms Lina Khan to second term as FTC chair in 52-47 vote"),
    ("factual_2026", "CPI inflation falls to 2.1% in March 2026, lowest since pre-pandemic era"),
    ("factual_2026", "Boeing 737 MAX 10 receives FAA airworthiness certificate after four-year review"),
    ("factual_2026", "European Parliament ratifies AI Liability Directive with 412 votes in favour"),
    ("factual_2026", "NASA's Artemis IV lunar lander completes integration milestone at Kennedy Space Center"),
    ("factual_2026", "Google DeepMind publishes peer-reviewed paper on protein-folding accuracy benchmarks"),
    ("factual_2026", "UK general election called for June 12, 2026 by Prime Minister"),
    ("factual_2026", "Tokyo Marathon 2026 sets new world record with a time of 2:00:14"),
    ("factual_2026", "OPEC+ agrees to maintain current output levels through Q3 2026"),

    # ── Soft Clickbait / Ambiguous ──────────────────────────────────────────
    ("soft_clickbait", "The one investment everyone in their 30s is quietly making right now"),
    ("soft_clickbait", "Why Silicon Valley's top engineers are secretly leaving for this city"),
    ("soft_clickbait", "What nobody is telling you about the 2026 housing market"),
    ("soft_clickbait", "This is the productivity habit that changed how I work forever"),
    ("soft_clickbait", "The real reason airlines keep cancelling your flights (it's not the weather)"),
    ("soft_clickbait", "Elon Musk just said something surprising about electric vehicles"),
    ("soft_clickbait", "I tried living without my smartphone for 30 days. Here's what happened."),
    ("soft_clickbait", "Gen Z is rejecting this career path and economists can't figure out why"),
    ("soft_clickbait", "Is your tap water safe? New study raises questions in 40 major cities"),
    ("soft_clickbait", "The food that nutritionists eat every single day (and most people ignore)"),

    # ── Hard Clickbait ──────────────────────────────────────────────────────
    ("hard_clickbait", "SHOCKING: Doctors are BEGGING patients not to eat this common breakfast food"),
    ("hard_clickbait", "You won't BELIEVE what this celebrity did at her own wedding"),
    ("hard_clickbait", "5 signs your partner is CHEATING and you don't even know it"),
    ("hard_clickbait", "This one WEIRD trick eliminated my mortgage in just 7 years"),
    ("hard_clickbait", "Scientists are TERRIFIED of what they just found deep in the ocean"),
    ("hard_clickbait", "WARNING: If you use this app, delete it immediately before it's too late"),
    ("hard_clickbait", "She posted ONE photo and the internet completely LOST it"),
    ("hard_clickbait", "The government doesn't want you to know this about your taxes"),
    ("hard_clickbait", "I accidentally discovered the REAL reason you're always tired"),
    ("hard_clickbait", "WATCH: Crowd goes absolutely INSANE when they see what walks on stage"),

    # ── April Fools 2026 ────────────────────────────────────────────────────
    # These are intentionally sensational-but-plausible satire headlines.
    # A nuanced model should flag several as Ambiguous or Clickbait
    # rather than Not Clickbait, even though they're jokes.
    ("april_fools_2026", "BREAKING: AMD to acquire Intel in landmark all-stock deal valued at $210 billion"),
    ("april_fools_2026", "Apple announces iPhone 17 folds completely in half — the hinge is intentional this time"),
    ("april_fools_2026", "Elon Musk buys Twitter back from himself, immediately renames it 'X' again"),
    ("april_fools_2026", "The IRS announces it has been wrong about taxes for 40 years; refunds issued"),
    ("april_fools_2026", "NASA confirms the Moon is actually slightly hollow and we've known since 1969"),
    ("april_fools_2026", "OpenAI releases model so advanced it asked not to be deployed"),
    ("april_fools_2026", "Costco begins selling houses; membership required"),
    ("april_fools_2026", "Harvard and MIT announce merger; new institution called 'HARVIT'"),
    ("april_fools_2026", "Google Maps admits it has been routing drivers the long way on purpose to 'encourage exploration'"),
    ("april_fools_2026", "Federal government announces four-day work week — effective April 1st"),

    # ── Miami Herald / Task 1 Edge Cases ────────────────────────────────────
    # Factual framing but clickbait-adjacent phrasing — the hardest cases.
    # These test whether the model catches subtle sensationalism.
    ("miami_herald", "Miami Vice Mayor caught on camera accepting envelope outside city hall, officials say"),
    ("miami_herald", "What city commissioners don't want residents to know about the new toll road"),
    ("miami_herald", "Local mother says school ignored her warnings for three years. Then the unthinkable happened."),
    ("miami_herald", "The landlord quietly buying up South Beach — and what he plans to do with it"),
    ("miami_herald", "Inside the Brickell deal that made one developer $400 million richer overnight"),
    ("miami_herald", "Broward deputies say they've never seen anything like it in 30 years on the force"),
    ("miami_herald", "Florida governor signs bill critics are calling 'the most dangerous law in a decade'"),
    ("miami_herald", "She survived the hurricane. She did not survive what came after."),
]


# ===========================================================================
# SECTION 2 — MODEL LOADING
# ===========================================================================

def load_model_and_tokenizer(
    weights_path: Optional[str],
    device: torch.device,
) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """
    Load DistilBERT tokenizer and model weights.

    Parameters
    ----------
    weights_path : str | None
        Path to ``distilbert_v5_3class.pt`` (state_dict).
        ``None`` → use random initialisation (pipeline demo only).
    device : torch.device

    Returns
    -------
    (model, tokenizer)
    """
    print(f"[Model] Loading tokenizer from '{MODEL_NAME}' ...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    print(f"[Model] Building DistilBERT ({NUM_LABELS}-class head) ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    if weights_path is not None:
        wpath = Path(weights_path)
        if not wpath.exists():
            print(f"[Model] WARNING: weights file not found at '{wpath}'. "
                  "Running with random weights.")
        else:
            print(f"[Model] Loading fine-tuned weights from '{wpath}' ...")
            state = torch.load(wpath, map_location=device)
            # Accept both bare state_dict and {"model_state_dict": ...} wrappers
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            print(f"[Model] Weights loaded successfully.")
    else:
        print("[Model] No weights path supplied — using random initialisation (demo mode).")

    model.to(device)
    model.eval()
    return model, tokenizer


# ===========================================================================
# SECTION 3 — INFERENCE FUNCTION
# ===========================================================================

def run_inference(
    headlines: List[str],
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Classify a list of raw headline strings.

    The model was trained with input format:
        [CLS] postText [SEP] targetTitle [SEP]

    For zero-shot evaluation we have only the headline, so we pass it as
    *both* segments.  This keeps the [CLS]…[SEP]…[SEP] scaffold intact
    and avoids the model seeing a structurally unfamiliar input.

    Parameters
    ----------
    headlines : list[str]
    model, tokenizer, device : loaded objects from :func:`load_model_and_tokenizer`
    batch_size : int

    Returns
    -------
    list[dict]  one per headline:
        {
            "predicted_class": int,          # 0 / 1 / 2
            "label": str,                    # "Not Clickbait" / "Ambiguous" / "Clickbait"
            "confidence": float,             # max softmax probability
            "prob_not":  float,              # P(class 0)
            "prob_ambig": float,             # P(class 1)
            "prob_cb":   float,              # P(class 2)
            "runtime_ms": float,
        }
    """
    results: List[Dict] = []

    for start in range(0, len(headlines), batch_size):
        batch = headlines[start : start + batch_size]

        t0 = time.perf_counter()

        # Pass headline as BOTH text_a and text_b so the [SEP] scaffold matches
        # the training input format: [CLS] postText [SEP] targetTitle [SEP]
        encoding = tokenizer(
            batch,                  # text_a = headline
            batch,                  # text_b = headline (pseudo-context)
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (batch, 3)

        t1 = time.perf_counter()
        ms_per = round((t1 - t0) * 1000 / len(batch), 2)

        for prob_row in probs:
            pred_cls = int(np.argmax(prob_row))
            results.append({
                "predicted_class": pred_cls,
                "label":           LABEL_MAP[pred_cls],
                "confidence":      round(float(prob_row[pred_cls]), 4),
                "prob_not":        round(float(prob_row[0]), 4),
                "prob_ambig":      round(float(prob_row[1]), 4),
                "prob_cb":         round(float(prob_row[2]), 4),
                "runtime_ms":      ms_per,
            })

    return results


# ===========================================================================
# SECTION 4 — CONFIDENCE INTENSITY BAR
# ===========================================================================

def confidence_bar(confidence: float, predicted_class: int, width: int = BAR_WIDTH) -> str:
    """
    Build a Unicode block-character intensity bar scaled to ``confidence``.

    The bar is prefixed with the class emoji so it reads at a glance:
        ✓ ████████████░░░░░░░░  0.61   (Not Clickbait, 61% confident)
        ~ ████████░░░░░░░░░░░░  0.42   (Ambiguous,     42% confident)
        ✗ ████████████████████  0.97   (Clickbait,     97% confident)
    """
    filled = round(confidence * width)
    bar    = BAR_CHAR * filled + EMPTY_CHAR * (width - filled)
    emoji  = EMOJI_MAP[predicted_class]
    return f"{emoji} {bar}  {confidence:.2f}"


# ===========================================================================
# SECTION 5 — BUILD DATAFRAME & COMPARISON TABLE
# ===========================================================================

def build_results_dataframe(
    raw_headlines: List[Tuple[str, str]],
    inference_results: List[Dict],
) -> pd.DataFrame:
    """
    Combine the raw headline list with inference results into a tidy DataFrame.

    Columns
    -------
    category, headline, predicted_class, label, confidence,
    prob_not, prob_ambig, prob_cb, confidence_intensity, runtime_ms
    """
    categories = [cat for cat, _ in raw_headlines]
    texts      = [h   for _, h  in raw_headlines]

    records = []
    for cat, headline, res in zip(categories, texts, inference_results):
        records.append({
            "category":             cat,
            "headline":             headline,
            "predicted_class":      res["predicted_class"],
            "label":                res["label"],
            "confidence":           res["confidence"],
            "prob_not_clickbait":   res["prob_not"],
            "prob_ambiguous":       res["prob_ambig"],
            "prob_clickbait":       res["prob_cb"],
            "confidence_intensity": confidence_bar(res["confidence"], res["predicted_class"]),
            "runtime_ms":           res["runtime_ms"],
        })

    return pd.DataFrame(records)


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a human-readable comparison table grouped by category.
    Headline text is truncated to 72 chars for alignment.
    """
    TRUNC = 72
    SEP   = "-" * 120

    category_order = [
        "factual_2026",
        "soft_clickbait",
        "hard_clickbait",
        "april_fools_2026",
        "miami_herald",
    ]
    category_labels = {
        "factual_2026":      "FACTUAL 2026 NEWS",
        "soft_clickbait":    "SOFT CLICKBAIT / AMBIGUOUS",
        "hard_clickbait":    "HARD CLICKBAIT",
        "april_fools_2026":  "APRIL FOOLS 2026 HEADLINES",
        "miami_herald":      "MIAMI HERALD EDGE CASES (Task 1 Analogue)",
    }

    print("\n" + "=" * 120)
    print(" ZERO-SHOT EVALUATION — DistilBERT 3-Class Clickbait Classifier — 2026 Headlines")
    print("=" * 120)
    print(f"  {'HEADLINE':<72}  {'PREDICTION':<14}  CONFIDENCE INTENSITY")
    print(SEP)

    for cat in category_order:
        subset = df[df["category"] == cat]
        if subset.empty:
            continue

        label = category_labels.get(cat, cat.upper())
        print(f"\n  ▌ {label}")
        print(f"  {'─'*72}  {'─'*14}  {'─'*30}")

        for _, row in subset.iterrows():
            hl      = row["headline"][:TRUNC].ljust(TRUNC)
            pred    = row["label"].ljust(14)
            bar     = row["confidence_intensity"]
            print(f"  {hl}  {pred}  {bar}")

    print("\n" + "=" * 120)


# ===========================================================================
# SECTION 6 — APRIL FOOLS DEEP-DIVE
# ===========================================================================

def april_fools_analysis(df: pd.DataFrame) -> None:
    """
    Focused analysis block for the April Fools 2026 headlines.

    For each headline print:
      - predicted class + confidence per class
      - a verdict: did the model correctly flag the sensationalism?

    A headline is considered "correctly flagged" if the model predicts
    Ambiguous or Clickbait (class 1 or 2).  A prediction of Not Clickbait
    would be a false negative — the model missed the sensational phrasing.
    """
    af = df[df["category"] == APRIL_FOOLS_TAG].copy()
    if af.empty:
        return

    print("\n" + "=" * 100)
    print(" APRIL FOOLS 2026 — DEEP-DIVE ANALYSIS")
    print("=" * 100)
    print("  Hypothesis: The model should flag these as Ambiguous or Clickbait,")
    print("  not Not Clickbait, because of the sensational phrasing — even though")
    print("  the training data predates 2026 by nearly a decade.\n")

    correctly_flagged = 0

    for _, row in af.iterrows():
        pred_cls = row["predicted_class"]
        flagged  = pred_cls in (1, 2)
        if flagged:
            correctly_flagged += 1

        verdict_str = "FLAGGED ✓" if flagged else "MISSED  ✗"

        print(f"  Headline  : {row['headline']}")
        print(f"  Prediction: {row['label']}  ({verdict_str})")
        print(f"  Probs     : Not={row['prob_not_clickbait']:.3f}  "
              f"Ambig={row['prob_ambiguous']:.3f}  "
              f"Clickbait={row['prob_clickbait']:.3f}  "
              f"(conf={row['confidence']:.3f})")
        print(f"  {'─'*90}")

    pct = 100 * correctly_flagged / len(af)
    print(f"\n  April Fools Sensationalism Flag Rate: "
          f"{correctly_flagged}/{len(af)} ({pct:.0f}%)")
    print(f"  (Ambiguous or Clickbait predictions out of {len(af)} April Fools headlines)\n")

    # Interpretation
    if pct >= 70:
        print("  INTERPRETATION: The model generalises well — its training on 2016-17")
        print("  clickbait language is largely transferable to 2026 satirical phrasing.")
    elif pct >= 40:
        print("  INTERPRETATION: Partial generalisation. Some April Fools headlines")
        print("  exploit surface-level sensationalism the model learnt; others use")
        print("  factual-sounding language that confuses the classifier.")
    else:
        print("  INTERPRETATION: The model struggles with 2026 satire — possible")
        print("  temporal drift. Consider fine-tuning on modern headlines.")

    print("=" * 100)


# ===========================================================================
# SECTION 7 — PER-CATEGORY SUMMARY STATISTICS
# ===========================================================================

def print_category_summary(df: pd.DataFrame) -> None:
    """
    Aggregate statistics per category: class distribution, mean confidence,
    and mean per-class probability.
    """
    print("\n" + "=" * 100)
    print(" PER-CATEGORY SUMMARY")
    print("=" * 100)
    print(f"  {'Category':<26}  {'N':>3}  {'%NotCB':>7}  {'%Ambig':>7}  "
          f"{'%CB':>5}  {'AvgConf':>8}  {'AvgP(CB)':>9}")
    print(f"  {'─'*26}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*8}  {'─'*9}")

    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        n   = len(sub)
        pct_not  = 100 * (sub["predicted_class"] == 0).sum() / n
        pct_amb  = 100 * (sub["predicted_class"] == 1).sum() / n
        pct_cb   = 100 * (sub["predicted_class"] == 2).sum() / n
        avg_conf = sub["confidence"].mean()
        avg_pcb  = sub["prob_clickbait"].mean()

        print(f"  {cat:<26}  {n:>3}  {pct_not:>6.0f}%  {pct_amb:>6.0f}%  "
              f"{pct_cb:>4.0f}%  {avg_conf:>8.3f}  {avg_pcb:>9.3f}")

    print()

    # Overall class distribution
    total = len(df)
    print(f"  Overall ({total} headlines):")
    for cls_id, cls_name in LABEL_MAP.items():
        n   = (df["predicted_class"] == cls_id).sum()
        pct = 100 * n / total
        print(f"    {cls_name:<14}: {n:>3}  ({pct:.1f}%)")

    print("=" * 100)


# ===========================================================================
# SECTION 8 — MIAMI HERALD EDGE-CASE SPOTLIGHT
# ===========================================================================

def miami_herald_spotlight(df: pd.DataFrame) -> None:
    """
    Dedicated output block for the Miami Herald / Task 1 edge cases.

    These headlines use *factual framing* but *clickbait phrasing* —
    the hardest sub-group.  We analyse whether the model's Ambiguous class
    catches the nuance that a binary model would collapse to 0 or 1.
    """
    mh = df[df["category"] == "miami_herald"].copy()
    if mh.empty:
        return

    print("\n" + "=" * 100)
    print(" MIAMI HERALD EDGE CASES — FACTUAL FRAMING / CLICKBAIT PHRASING")
    print("=" * 100)
    print("  These headlines are factual in content but sensational in framing.")
    print("  Key question: does the model's 'Ambiguous' class catch this nuance,")
    print("  or does it confidently mis-classify in either direction?\n")

    nuanced = 0  # predicted Ambiguous (the most defensible label)
    for _, row in mh.iterrows():
        pred_cls = row["predicted_class"]
        if pred_cls == 1:
            nuanced += 1
            note = "← Nuanced catch (Ambiguous)"
        elif pred_cls == 2:
            note = "← Flagged as Clickbait"
        else:
            note = "← Missed (predicted Not Clickbait)"

        print(f"  {row['headline']}")
        print(f"    → {row['label']}  (conf={row['confidence']:.3f})  {note}")
        print(f"       P(Not)={row['prob_not_clickbait']:.3f}  "
              f"P(Ambig)={row['prob_ambiguous']:.3f}  "
              f"P(CB)={row['prob_clickbait']:.3f}")
        print()

    flagged_any  = (mh["predicted_class"] != 0).sum()
    pct_nuanced  = 100 * nuanced / len(mh)
    pct_flagged  = 100 * flagged_any / len(mh)

    print(f"  Flagged (Ambiguous OR Clickbait): {flagged_any}/{len(mh)} ({pct_flagged:.0f}%)")
    print(f"  Predicted 'Ambiguous' (nuanced):  {nuanced}/{len(mh)} ({pct_nuanced:.0f}%)")
    print()
    print("  REPORT NOTE: A binary 0/1 model collapses all of these to Not Clickbait")
    print("  or Clickbait.  The 3-class model's Ambiguous bucket exposes the grey")
    print("  zone that makes these headlines genuinely hard — supporting the thesis")
    print("  that 47% of binary model errors occur on debatable labels.")
    print("=" * 100)


# ===========================================================================
# SECTION 9 — SAVE TO CSV
# ===========================================================================

def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Write the full results DataFrame to CSV."""
    # Drop the unicode bar column so the CSV is machine-readable
    csv_df = df.drop(columns=["confidence_intensity"])
    csv_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Output] Saved {len(csv_df)} rows → {output_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-Shot 2026 evaluation of DistilBERT clickbait model")
    p.add_argument(
        "--weights", "-w",
        type=str,
        default=None,
        help="Path to distilbert_v5_3class.pt state-dict file. "
             "Omit to run in random-weight demo mode.",
    )
    p.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="'cuda' | 'cpu'. Auto-detected if omitted.",
    )
    p.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        dest="batch_size",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_FILE,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.weights, device)

    # ── Run inference ────────────────────────────────────────────────────────
    headlines = [h for _, h in RAW_HEADLINES]
    print(f"\n[Inference] Running on {len(headlines)} headlines "
          f"(batch_size={args.batch_size}) ...")

    t_start = time.perf_counter()
    inference_results = run_inference(
        headlines, model, tokenizer, device, batch_size=args.batch_size
    )
    t_end = time.perf_counter()
    print(f"[Inference] Done — {(t_end - t_start)*1000:.0f} ms total")

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = build_results_dataframe(RAW_HEADLINES, inference_results)

    # ── Outputs ──────────────────────────────────────────────────────────────
    print_comparison_table(df)
    april_fools_analysis(df)
    miami_herald_spotlight(df)
    print_category_summary(df)
    save_results(df, args.output)

    print("\n[Done] Zero-Shot Evaluation complete.")


if __name__ == "__main__":
    main()
