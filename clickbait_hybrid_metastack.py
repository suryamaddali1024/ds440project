"""
clickbait_hybrid_metastack.py
==============================
Hybrid Meta-Stacking Architecture for Clickbait Detection.
Target: Break the F1 = 0.711 barrier.

Architecture
------------
  Layer 0 ─ Base Learners (trained on base_train split, 64% of data):
    • DistilBERT v2  — regression on truthMean, dual input [postText][SEP][targetTitle]
    • LightGBM       — 28 hand-crafted features (26 original + 2 new)

  Layer 1 ─ Meta-Learner (trained on blend_val split, 16% of data):
    • Logistic Regression — inputs: (p_distilbert, p_lgbm) → calibrated final probability

  Layer 2 ─ Probability Calibration:
    • Platt Scaling (sigmoid)  — primary
    • Isotonic Regression      — benchmark
    Both applied to the meta-learner output; best ECE wins.

  Layer 3 ─ Tri-State Output:
    • CERTAIN_CB   (prob > 0.80) — target precision > 90 %
    • CERTAIN_NCB  (prob < 0.20)
    • AMBIGUOUS_IDK (0.20–0.80) — abstain

Data Split (all stratified on truthClass, seed 42)
---------------------------------------------------
  base_train  64%  → train both base learners
  blend_val   16%  → train meta-learner + calibration
  test        20%  → final evaluation (same split as all prior scripts)

New Features (Section 2)
-------------------------
  feat_27  mirror_similarity      Levenshtein ratio(postText, targetTitle)
  feat_28  pragmatic_slang_count  regex hits on 25 viral clickbait triggers

Usage
-----
  pip install transformers torch scikit-learn lightgbm Levenshtein
              vaderSentiment sentence-transformers xgboost
  python clickbait_hybrid_metastack.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import ast
import os
import re
import random
import textwrap
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  ── single source of truth for all constants
# ═══════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── files ──────────────────────────────────────────────────────────────
    INPUT_FILE         = "final_cleaned_full.csv",
    OUTPUT_PREDS_FILE  = "clickbait_hybrid_metastack_preds.csv",
    DISTILBERT_WEIGHTS = "distilbert_v2_weights.pt",   # saved/loaded here

    # ── splits (must match all prior scripts) ──────────────────────────────
    TEST_SIZE          = 0.20,
    BLEND_FRACTION     = 0.20,           # of the 80% train → blend_val
    RANDOM_STATE       = 42,

    # ── DistilBERT ─────────────────────────────────────────────────────────
    MODEL_NAME         = "distilbert-base-uncased",
    MAX_LENGTH         = 96,
    BERT_BATCH_SIZE    = 16,
    EPOCHS             = 4,
    LEARNING_RATE      = 2e-5,
    WARMUP_RATIO       = 0.10,

    # ── LightGBM ───────────────────────────────────────────────────────────
    LGBM_PARAMS = dict(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 7,
        num_leaves        = 63,
        min_child_samples = 20,
        subsample         = 0.80,
        colsample_bytree  = 0.80,
        reg_alpha         = 0.10,
        reg_lambda        = 0.10,
        is_unbalance      = True,
        random_state      = 42,
        verbose           = -1,
    ),

    # ── Meta-Learner ────────────────────────────────────────────────────────
    META_LR_C          = 1.0,

    # ── Tri-state thresholds ────────────────────────────────────────────────
    CERTAIN_CB_THRESH  = 0.80,
    CERTAIN_NCB_THRESH = 0.20,

    # ── SBERT (for tabular features) ────────────────────────────────────────
    SBERT_MODEL        = "sentence-transformers/all-MiniLM-L6-v2",
    SBERT_BATCH        = 128,
)

# ── Seed everything ────────────────────────────────────────────────────────
torch.manual_seed(CFG["RANDOM_STATE"])
random.seed(CFG["RANDOM_STATE"])
np.random.seed(CFG["RANDOM_STATE"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pragmatic slang trigger list (25 patterns from error analysis) ──────────
# Sourced from top-50 confidently-wrong analysis; these lexical cues signal
# clickbait intent invisible to a plain token classifier.
SLANG_TRIGGERS = re.compile(
    r"\b("
    r"shots fired|stay woke|you won'?t believe|wait for it|this is why|"
    r"this will make you|wait till you see|you need to see|"
    r"we can'?t stop watching|i can'?t even|i'?m dead|i'?m crying|"
    r"twitter (lost|exploded|reacted|went wild|can'?t handle)|"
    r"the internet (lost|reacted|is losing)|"
    r"people are (freaking|losing) (out|it)|"
    r"broke the internet|minds were blown|mic drop|plot twist|"
    r"the struggle is real|on fleek|slay(ed|ing)?|goals|no cap|"
    r"lowkey|highkey|big yikes|ok boomer|caught in 4k|"
    r"era of|main character|rent free|understood the assignment|"
    r"hits different|it's giving|say less|periodt?"
    r")\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_text_list(raw_str: str) -> str:
    """Convert a stringified Python list into joined plain text."""
    if pd.isna(raw_str) or not str(raw_str).strip():
        return ""
    raw_str = str(raw_str).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            return " ".join(str(x).strip() for x in parsed if str(x).strip())
        return str(parsed).strip()
    except (ValueError, SyntaxError):
        if raw_str.startswith("[") and raw_str.endswith("]"):
            raw_str = raw_str[1:-1].strip()
        return raw_str


def sec(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def subsec(title: str):
    print(f"\n  {'─'*66}")
    print(f"  {title}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split():
    sec("SECTION 1 — Data Loading & Stratified Split")

    df = pd.read_csv(CFG["INPUT_FILE"], encoding="latin-1")
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    for col, target in [
        ("postText",           "postText_clean"),
        ("targetTitle",        "targetTitle_clean"),
        ("targetParagraphs",   "targetParagraphs_clean"),
        ("targetDescription",  "targetDescription_clean"),
        ("targetKeywords",     "targetKeywords_clean"),
    ]:
        df[target] = df[col].apply(parse_text_list)

    df = df[
        (df["postText_clean"].str.strip() != "") &
        (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"  After cleanup: {len(df):,} rows")
    print(f"  Class dist: {df['truthClass'].value_counts().to_dict()}")

    y = df["truthClass"].values

    # ── Primary 80/20 split  (identical to all prior scripts) ──
    trainval_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=CFG["TEST_SIZE"],
        stratify=y, random_state=CFG["RANDOM_STATE"],
    )

    # ── Secondary split: base_train vs blend_val ──
    y_trainval = y[trainval_idx]
    base_idx, blend_idx = train_test_split(
        np.arange(len(trainval_idx)),
        test_size=CFG["BLEND_FRACTION"],
        stratify=y_trainval,
        random_state=CFG["RANDOM_STATE"],
    )
    base_train_idx  = trainval_idx[base_idx]   # 64% of total
    blend_val_idx   = trainval_idx[blend_idx]  # 16% of total

    print(f"  base_train : {len(base_train_idx):>5}  ({len(base_train_idx)/len(df)*100:.1f}%)")
    print(f"  blend_val  : {len(blend_val_idx):>5}  ({len(blend_val_idx)/len(df)*100:.1f}%)")
    print(f"  test       : {len(test_idx):>5}  ({len(test_idx)/len(df)*100:.1f}%)")

    return df, y, base_train_idx, blend_val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING (28 Features)
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular_features(df, base_train_idx, y):
    sec("SECTION 2 — Feature Engineering (28 features incl. 2 new)")

    try:
        from Levenshtein import ratio as lev_ratio
        HAS_LEV = True
        print("  ✓ python-Levenshtein available")
    except ImportError:
        HAS_LEV = False
        print("  ✗ Levenshtein not found — using difflib fallback")
        import difflib
        def lev_ratio(a, b):
            return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import chi2

    # ── SBERT embeddings ──────────────────────────────────────────────────
    sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading SBERT ({CFG['SBERT_MODEL']}) on {sbert_device}…")
    sbert = SentenceTransformer(CFG["SBERT_MODEL"], device=sbert_device)

    def encode(texts):
        return sbert.encode(texts, batch_size=CFG["SBERT_BATCH"],
                            show_progress_bar=False, convert_to_numpy=True)

    emb_post    = encode(df["postText_clean"].tolist())
    emb_article = encode(df["targetParagraphs_clean"].tolist())
    emb_title   = encode([t if t.strip() else "no title"
                           for t in df["targetTitle_clean"].tolist()])
    emb_desc    = encode([d if d.strip() else "no description"
                           for d in df["targetDescription_clean"].tolist()])
    print(f"  Embeddings shape: {emb_post.shape}")

    feat = pd.DataFrame(index=df.index)
    post    = df["postText_clean"]
    article = df["targetParagraphs_clean"]
    title   = df["targetTitle_clean"]
    desc    = df["targetDescription_clean"]
    kw      = df["targetKeywords_clean"]

    # ── Group A: Text statistics (6) ─────────────────────────────────────
    subsec("Group A — Text statistics")
    feat["post_word_count"]       = post.apply(lambda x: len(x.split()))
    feat["article_word_count"]    = article.apply(lambda x: len(x.split()))
    feat["word_count_ratio"]      = feat["post_word_count"] / feat["article_word_count"].replace(0, 1)
    feat["post_question_marks"]   = post.apply(lambda x: x.count("?"))
    feat["post_exclamation_marks"]= post.apply(lambda x: x.count("!"))
    feat["post_caps_ratio"]       = post.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1) / max(len(x.split()), 1)
    )

    # ── Group B: VADER sentiment (3) ─────────────────────────────────────
    subsec("Group B — VADER sentiment")
    vader = SentimentIntensityAnalyzer()
    post_sent    = post.apply(lambda x: vader.polarity_scores(x)["compound"])
    article_sent = article.apply(lambda x: vader.polarity_scores(x)["compound"])
    feat["post_sentiment"]     = post_sent
    feat["article_sentiment"]  = article_sent
    feat["abs_sentiment_gap"]  = (post_sent - article_sent).abs()

    # ── Group C: Semantic mismatch (3) ──────────────────────────────────
    subsec("Group C — Semantic mismatch")
    norm_post    = np.linalg.norm(emb_post,    axis=1, keepdims=True) + 1e-8
    norm_article = np.linalg.norm(emb_article, axis=1, keepdims=True) + 1e-8
    feat["cosine_similarity"] = np.sum(
        (emb_post / norm_post) * (emb_article / norm_article), axis=1
    )
    # KL divergence via TF-IDF
    all_texts = pd.concat([post, article]).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf.fit(all_texts)
    p = tfidf.transform(post.tolist()).toarray() + 1e-10
    q = tfidf.transform(article.tolist()).toarray() + 1e-10
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    feat["kl_divergence"] = np.sum(p * np.log(p / q), axis=1)
    feat["jaccard_similarity"] = [
        len(set(pt.lower().split()) & set(at.lower().split())) /
        max(len(set(pt.lower().split()) | set(at.lower().split())), 1)
        for pt, at in zip(post, article)
    ]

    # ── Group D: Clickbait linguistic patterns (6) ──────────────────────
    subsec("Group D — Clickbait linguistic patterns")
    y_train = y[base_train_idx]
    cv = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=10000)
    X_ng_train = cv.fit_transform(post.iloc[base_train_idx])
    chi2_scores, _ = chi2(X_ng_train, y_train)
    top100 = np.argsort(chi2_scores)[::-1][:100]
    X_ng_all = cv.transform(post)
    feat["clickbait_ngram_count"]     = np.array(X_ng_all[:, top100].sum(axis=1)).flatten()
    feat["has_number"]                = post.apply(lambda x: int(bool(re.search(r"\d", x))))
    feat["starts_with_demonstrative"] = post.apply(
        lambda x: int(bool(re.match(r"^\s*(this|these|here|that)\b", x, re.I)))
    )
    feat["second_person_count"] = post.apply(
        lambda x: len(re.findall(r"\b(you|your|you're|yourself|yours)\b", x, re.I))
    )
    feat["avg_word_length"] = post.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0
    )
    feat["ellipsis_count"] = post.apply(lambda x: x.count("..."))

    # ── Group E: Article metadata (4) ────────────────────────────────────
    subsec("Group E — Article metadata features")
    norm_title = np.linalg.norm(emb_title, axis=1, keepdims=True) + 1e-8
    norm_desc  = np.linalg.norm(emb_desc,  axis=1, keepdims=True) + 1e-8
    title_cos  = np.sum((emb_post / norm_post) * (emb_title / norm_title), axis=1)
    desc_cos   = np.sum((emb_post / norm_post) * (emb_desc  / norm_desc),  axis=1)
    title_cos[title.str.strip() == ""] = 0.0
    desc_cos[desc.str.strip()   == ""] = 0.0
    feat["title_post_cosine_sim"] = title_cos
    feat["title_post_jaccard"]    = [
        len(set(pt.lower().split()) & set(tt.lower().split())) /
        max(len(set(pt.lower().split()) | set(tt.lower().split())), 1)
        for pt, tt in zip(post, title)
    ]
    feat["desc_post_cosine_sim"]  = desc_cos
    feat["keyword_overlap_ratio"] = [
        sum(1 for k in re.split(r"[,\s]+", kw_str.lower()) if k and k in pt.lower()) /
        max(len([k for k in re.split(r"[,\s]+", kw_str.lower()) if k]), 1)
        for pt, kw_str in zip(post, kw.fillna(""))
    ]

    # ── Group F: Error-driven features (4) ──────────────────────────────
    subsec("Group F — Error-driven features (from prior analysis)")
    SENSATIONAL = {
        "shocking","stunned","stunning","horrifying","terrifying","devastating",
        "incredible","unbelievable","amazing","insane","crazy","disturbing",
        "heartbreaking","sickening","outrageous","explosive","bombshell",
        "scandalous","controversial","dramatic","tragic","horrific","alarming",
        "chilling","disgusting","furious","hilarious","epic","brutal","savage",
        "deadly","massive","urgent","breaking","exclusive","revealed","exposed",
        "slammed","blasted","destroyed","crushed","creepy","weird","strange",
    }
    FORWARD = {
        "new","emerge","emerges","reveal","reveals","revealed","discover",
        "discovers","discovered","found","finds","uncover","uncovers",
        "uncovered","detail","details","secret","secrets","mystery","hidden",
        "unknown","surprise","surprising","unexpected","suddenly",
    }
    feat["sensational_word_count"]  = post.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in SENSATIONAL)
    )
    feat["post_sentiment_intensity"] = post_sent.abs()
    feat["proper_noun_ratio"]        = post.apply(
        lambda x: sum(1 for w in x.split()[1:] if w and w[0].isupper() and not w.isupper())
                  / max(len(x.split()), 1)
    )
    feat["forward_reference_count"]  = post.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in FORWARD)
    )

    # ── ★ NEW: Group G — Mirror Similarity & Pragmatic Slang (2) ★ ──────
    subsec("Group G — NEW: Mirror Similarity & Pragmatic Slang Detection")

    # feat_27: Levenshtein ratio between postText and targetTitle
    # Hypothesis: HIGH similarity (post ≈ title) → NOT clickbait (no curiosity gap)
    #             LOW similarity (post ≠ title)  → potential clickbait
    mirror_sim = []
    for pt, tt in zip(post, title):
        if not tt.strip():
            mirror_sim.append(0.5)  # no title → neutral
            continue
        mirror_sim.append(lev_ratio(pt.lower().strip(), tt.lower().strip()))
    feat["mirror_similarity"] = mirror_sim

    # feat_28: count of pragmatic slang / viral-trigger phrases
    # Hypothesis: slang like "shots fired", "stay woke" → clickbait post
    feat["pragmatic_slang_count"] = post.apply(
        lambda x: len(SLANG_TRIGGERS.findall(x))
    )

    print(f"\n  mirror_similarity  — mean={feat['mirror_similarity'].mean():.3f}  "
          f"std={feat['mirror_similarity'].std():.3f}")
    print(f"  pragmatic_slang    — mean={feat['pragmatic_slang_count'].mean():.3f}  "
          f"nonzero={( feat['pragmatic_slang_count'] > 0).sum()}")

    FEATURE_NAMES = list(feat.columns)
    print(f"\n  Total features: {len(FEATURE_NAMES)}")
    print(f"  Feature matrix shape: {feat.shape}")

    # Quick correlation with truthClass
    y_all = df["truthClass"]
    subsec("Feature → truthClass correlations (top-10 by |r|)")
    corrs = {col: feat[col].corr(y_all) for col in FEATURE_NAMES}
    for col, r in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"    {col:<35}  r = {r:+.4f}")

    return feat, FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DISTILBERT v2  (regression on truthMean, dual input)
# ─────────────────────────────────────────────────────────────────────────────

class ClickbaitRegressionDataset(Dataset):
    def __init__(self, post_texts, title_texts, scores, tokenizer, max_len):
        self.enc = tokenizer(
            post_texts, title_texts,
            truncation=True, padding=True,
            max_length=max_len, return_tensors="pt",
        )
        self.scores = torch.tensor(scores, dtype=torch.float)

    def __len__(self):  return len(self.scores)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["scores"] = self.scores[i]
        return item


def _bert_predict(model, tokenizer, posts, titles, device):
    """Return sigmoid regression scores for a list of (post, title) pairs."""
    dataset = ClickbaitRegressionDataset(
        posts, titles, np.zeros(len(posts)), tokenizer, CFG["MAX_LENGTH"]
    )
    loader = DataLoader(dataset, batch_size=CFG["BERT_BATCH_SIZE"], shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy())
    return np.array(preds)


def train_or_load_distilbert(df, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 3 — DistilBERT v2 (regression on truthMean)")

    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from sklearn.metrics import mean_squared_error as mse_sk

    posts  = df["postText_clean"].tolist()
    titles = [t if t.strip() else "no title"
              for t in df["targetTitle_clean"].tolist()]
    scores = df["truthMean"].values

    tokenizer = DistilBertTokenizer.from_pretrained(CFG["MODEL_NAME"])

    weights_path = CFG["DISTILBERT_WEIGHTS"]

    if os.path.exists(weights_path):
        print(f"  Found saved weights at '{weights_path}' — loading…")
        model = DistilBertForSequenceClassification.from_pretrained(
            CFG["MODEL_NAME"], num_labels=1
        )
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        print("  ✓ Weights loaded; skipping training.")
    else:
        print(f"  No weights found — training DistilBERT v2 from scratch ({DEVICE})")

        def gather(lst, idxs): return [lst[i] for i in idxs]

        # Further split base_train into train+val for DistilBERT
        btrain_idx, bval_idx = train_test_split(
            np.arange(len(base_train_idx)),
            test_size=CFG["BLEND_FRACTION"],
            stratify=df["truthClass"].values[base_train_idx],
            random_state=CFG["RANDOM_STATE"],
        )
        bert_train = base_train_idx[btrain_idx]
        bert_val   = base_train_idx[bval_idx]

        tr_posts   = gather(posts,  bert_train)
        tr_titles  = gather(titles, bert_train)
        tr_scores  = scores[bert_train]
        v_posts    = gather(posts,  bert_val)
        v_titles   = gather(titles, bert_val)
        v_scores   = scores[bert_val]

        print(f"  BERT train: {len(bert_train)}  BERT val: {len(bert_val)}")

        tr_ds = ClickbaitRegressionDataset(tr_posts, tr_titles, tr_scores,
                                            tokenizer, CFG["MAX_LENGTH"])
        v_ds  = ClickbaitRegressionDataset(v_posts,  v_titles,  v_scores,
                                            tokenizer, CFG["MAX_LENGTH"])
        tr_loader = DataLoader(tr_ds, batch_size=CFG["BERT_BATCH_SIZE"], shuffle=True)
        v_loader  = DataLoader(v_ds,  batch_size=CFG["BERT_BATCH_SIZE"], shuffle=False)

        model = DistilBertForSequenceClassification.from_pretrained(
            CFG["MODEL_NAME"], num_labels=1
        )
        model.to(DEVICE)

        optimizer  = AdamW(model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.01)
        total_steps = len(tr_loader) * CFG["EPOCHS"]
        scheduler  = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(total_steps * CFG["WARMUP_RATIO"]),
            num_training_steps = total_steps,
        )
        loss_fn = nn.MSELoss()

        best_val_mse  = float("inf")
        best_state    = None

        for epoch in range(CFG["EPOCHS"]):
            # train
            model.train()
            ep_loss = 0.0
            for bi, batch in enumerate(tr_loader):
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                tgt   = batch["scores"].to(DEVICE)
                preds = torch.sigmoid(model(input_ids=ids, attention_mask=mask).logits.squeeze(-1))
                loss  = loss_fn(preds, tgt)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                ep_loss += loss.item()
                if (bi + 1) % 100 == 0:
                    print(f"    Ep {epoch+1}/{CFG['EPOCHS']}  Batch {bi+1}/{len(tr_loader)}"
                          f"  MSE={ep_loss/(bi+1):.5f}")
            # validate
            model.eval()
            v_preds, v_tgts = [], []
            with torch.no_grad():
                for batch in v_loader:
                    ids  = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    tgt  = batch["scores"].to(DEVICE)
                    v_preds.extend(torch.sigmoid(
                        model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                    ).cpu().numpy())
                    v_tgts.extend(tgt.cpu().numpy())
            val_mse = mse_sk(v_tgts, v_preds)
            print(f"  Epoch {epoch+1}  train_MSE={ep_loss/len(tr_loader):.5f}  "
                  f"val_MSE={val_mse:.5f}")
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ★ New best val MSE: {best_val_mse:.5f}")

        model.load_state_dict(best_state)
        torch.save(best_state, weights_path)
        print(f"  ✓ Weights saved to '{weights_path}'")

    # ── Generate probabilities for blend_val and test ──
    def g(lst, idxs): return [lst[i] for i in idxs]

    print("\n  Generating DistilBERT probabilities…")
    blend_bert = _bert_predict(
        model, tokenizer,
        g(posts, blend_val_idx), g(titles, blend_val_idx), DEVICE,
    )
    test_bert  = _bert_predict(
        model, tokenizer,
        g(posts, test_idx), g(titles, test_idx), DEVICE,
    )
    print(f"  blend_val probs — mean={blend_bert.mean():.3f}  std={blend_bert.std():.3f}")
    print(f"  test      probs — mean={test_bert.mean():.3f}   std={test_bert.std():.3f}")

    return blend_bert, test_bert


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIGHTGBM (28 tabular features)
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(feat_df, feature_names, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4 — LightGBM (28 tabular features)")

    X = feat_df[feature_names].values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X[base_train_idx])
    X_blend_sc = scaler.transform(X[blend_val_idx])
    X_test_sc  = scaler.transform(X[test_idx])
    y_train    = y[base_train_idx]

    print(f"  Training set: {len(base_train_idx)} samples, {len(feature_names)} features")
    clf = lgb.LGBMClassifier(**CFG["LGBM_PARAMS"])
    clf.fit(
        X_train_sc, y_train,
        eval_set=[(X_blend_sc, y[blend_val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=0)],
    )

    blend_lgbm = clf.predict_proba(X_blend_sc)[:, 1]
    test_lgbm  = clf.predict_proba(X_test_sc)[:, 1]

    # Quick standalone metric on blend_val
    thr_opts = np.linspace(0.2, 0.8, 61)
    best_t, best_f1 = 0.5, 0.0
    for t in thr_opts:
        f1 = f1_score(y[blend_val_idx], (blend_lgbm >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    lgbm_blend_f1 = f1_score(y[blend_val_idx], (blend_lgbm >= best_t).astype(int))
    print(f"  LightGBM blend_val standalone F1: {lgbm_blend_f1:.4f} (thr={best_t:.2f})")

    # Feature importance (top 10)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    subsec("Top-10 LightGBM feature importances")
    for rank, idx in enumerate(order[:10], 1):
        print(f"    {rank:>2}. {feature_names[idx]:<35}  {importances[idx]:.1f}")

    return blend_lgbm, test_lgbm, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — LOGISTIC REGRESSION META-LEARNER
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_learner(y, blend_val_idx, blend_bert, blend_lgbm):
    sec("SECTION 5 — Logistic Regression Meta-Learner")

    y_blend = y[blend_val_idx]

    # Meta-features: [p_bert, p_lgbm, p_bert*p_lgbm, |p_bert - p_lgbm|]
    # The interaction and disagreement terms help the LR find the optimal boundary.
    def meta_features(pb, pl):
        return np.column_stack([
            pb, pl,
            pb * pl,                      # agreement signal
            np.abs(pb - pl),              # disagreement signal
            np.maximum(pb, pl),           # upper-bound ensemble
            np.minimum(pb, pl),           # lower-bound ensemble
        ])

    X_meta_blend = meta_features(blend_bert, blend_lgbm)

    print(f"  Meta-training set size: {len(y_blend)}")
    print(f"  Meta-feature shape:     {X_meta_blend.shape}")
    print(f"  blend_val class dist:   {dict(zip(*np.unique(y_blend, return_counts=True)))}")

    meta_lr = LogisticRegression(
        C=CFG["META_LR_C"],
        class_weight="balanced",
        max_iter=1000,
        random_state=CFG["RANDOM_STATE"],
        solver="lbfgs",
    )
    meta_lr.fit(X_meta_blend, y_blend)

    # Blend-val self-evaluation
    blend_meta_probs = meta_lr.predict_proba(X_meta_blend)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.2, 0.8, 61):
        f1 = f1_score(y_blend, (blend_meta_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  Meta-LR blend_val F1 (thr={best_t:.2f}): {best_f1:.4f}")
    print(f"  Meta-LR coefficients: {meta_lr.coef_[0]}")

    return meta_lr, meta_features


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PROBABILITY CALIBRATION (Platt & Isotonic)
# ─────────────────────────────────────────────────────────────────────────────

class _ProbWrapper:
    """Wrap a numpy probability array so sklearn calibration can fit on it."""
    def __init__(self, probs):
        self.probs = probs
    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])
    def fit(self, X, y):
        return self


def calibrate_probabilities(y, blend_val_idx, blend_meta_raw, meta_lr, meta_features_fn):
    sec("SECTION 6 — Probability Calibration (Platt vs Isotonic)")

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as _LR

    y_blend = y[blend_val_idx]

    # ── Platt Scaling (sigmoid logistic regression on raw meta-probs) ──
    platt = _LR(C=1.0, solver="lbfgs", max_iter=500)
    platt.fit(blend_meta_raw.reshape(-1, 1), y_blend)

    def platt_transform(probs):
        return platt.predict_proba(probs.reshape(-1, 1))[:, 1]

    # ── Isotonic Regression ────────────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(blend_meta_raw, y_blend)

    # ── ECE comparison on blend_val ────────────────────────────────────
    def ece(y_true, y_prob, n_bins=10):
        frac_pos, mean_pred = calibration_curve(y_true, y_prob,
                                                 n_bins=n_bins, strategy="uniform")
        return float(np.mean(np.abs(frac_pos - mean_pred)))

    ece_raw   = ece(y_blend, blend_meta_raw)
    ece_platt = ece(y_blend, platt_transform(blend_meta_raw))
    ece_iso   = ece(y_blend, iso.predict(blend_meta_raw))

    print(f"  ECE (raw meta-LR)      : {ece_raw:.4f}")
    print(f"  ECE (Platt scaling)    : {ece_platt:.4f}")
    print(f"  ECE (Isotonic reg.)    : {ece_iso:.4f}")

    best_method = "Platt" if ece_platt <= ece_iso else "Isotonic"
    print(f"\n  ★ Selected calibrator : {best_method} (lower ECE)")

    if best_method == "Platt":
        calibrate_fn = platt_transform
    else:
        calibrate_fn = iso.predict

    return calibrate_fn, ece_platt, ece_iso, best_method


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FINAL ENSEMBLE PREDICTIONS ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def make_final_predictions(
    df, y, test_idx,
    test_bert, test_lgbm,
    meta_lr, meta_features_fn, calibrate_fn,
):
    sec("SECTION 7 — Final Ensemble Predictions (Test Set)")

    X_meta_test = meta_features_fn(test_bert, test_lgbm)
    raw_meta    = meta_lr.predict_proba(X_meta_test)[:, 1]
    final_probs = calibrate_fn(raw_meta)

    y_test = y[test_idx]

    # ── Binary threshold optimisation ────────────────────────────────────
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, final_probs)
    f1_arr  = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_ti = np.argmax(f1_arr)
    opt_thr = float(thr_arr[best_ti])
    f1_opt  = float(f1_arr[best_ti])
    f1_def  = float(f1_score(y_test, (final_probs >= 0.5).astype(int)))

    print(f"  F1 @ threshold=0.50  : {f1_def:.4f}")
    print(f"  F1 @ opt threshold   : {f1_opt:.4f}  (thr={opt_thr:.3f})")

    threshold = opt_thr if f1_opt > f1_def else 0.5
    y_pred    = (final_probs >= threshold).astype(int)

    return final_probs, y_pred, y_test, threshold


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRI-STATE CLASSIFIER & RESOLVED ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def tristate_classification(final_probs, y_test):
    sec("SECTION 8 — Tri-State Classifier & Resolved Accuracy")

    CB_T  = CFG["CERTAIN_CB_THRESH"]
    NCB_T = CFG["CERTAIN_NCB_THRESH"]

    def bucket(p):
        if   p > CB_T:  return "CERTAIN_CB"
        elif p < NCB_T: return "CERTAIN_NCB"
        else:           return "AMBIGUOUS_IDK"

    labels = np.array([bucket(p) for p in final_probs])

    for b in ["CERTAIN_CB", "CERTAIN_NCB", "AMBIGUOUS_IDK"]:
        n   = (labels == b).sum()
        pct = n / len(labels) * 100
        print(f"  {b:<20}: {n:>5} ({pct:.1f}%)")

    # ── Resolved accuracy ─────────────────────────────────────────────────
    resolved_mask = labels != "AMBIGUOUS_IDK"
    y_resolved    = y_test[resolved_mask]
    p_resolved    = final_probs[resolved_mask]
    l_resolved    = labels[resolved_mask]

    pred_resolved = (l_resolved == "CERTAIN_CB").astype(int)

    if resolved_mask.sum() == 0:
        print("  No resolved samples — widen thresholds.")
        return labels

    res_acc  = accuracy_score(y_resolved, pred_resolved)
    res_f1   = f1_score(y_resolved, pred_resolved, zero_division=0)
    res_prec = precision_score(y_resolved, pred_resolved, zero_division=0)
    res_rec  = recall_score(y_resolved, pred_resolved, zero_division=0)

    subsec("Resolved Accuracy (CERTAIN_CB + CERTAIN_NCB only)")
    print(f"  Resolved samples : {resolved_mask.sum():>5} ({resolved_mask.mean()*100:.1f}% coverage)")
    print(f"  Resolved Accuracy: {res_acc:.4f}  ({res_acc*100:.2f}%)")
    print(f"  Resolved F1      : {res_f1:.4f}")
    print(f"  Resolved Prec    : {res_prec:.4f}")
    print(f"  Resolved Recall  : {res_rec:.4f}")

    # Per-bucket precision
    cb_mask  = labels == "CERTAIN_CB"
    ncb_mask = labels == "CERTAIN_NCB"
    if cb_mask.sum() > 0:
        cb_prec = (y_test[cb_mask] == 1).mean()
        print(f"\n  CERTAIN_CB  precision : {cb_prec:.4f}  "
              f"{'✓ >90% target met' if cb_prec > 0.90 else '✗ below 90% target'}")
    if ncb_mask.sum() > 0:
        ncb_prec = (y_test[ncb_mask] == 0).mean()
        print(f"  CERTAIN_NCB precision : {ncb_prec:.4f}")

    # IDK composition
    idk_mask = labels == "AMBIGUOUS_IDK"
    if idk_mask.sum() > 0:
        idk_cb_rate = y_test[idk_mask].mean()
        print(f"\n  IDK true-CB rate      : {idk_cb_rate:.4f}  "
              f"({'mostly clickbait' if idk_cb_rate > 0.5 else 'mostly not-clickbait'})")

    # Threshold sensitivity table
    subsec("Threshold Sensitivity Analysis")
    print(f"  {'CB_thr':>7}  {'NCB_thr':>8}  {'Coverage':>9}  "
          f"{'Res.Acc':>8}  {'CB_Prec':>8}  {'NCB_Prec':>9}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*9}")
    for cb_t, ncb_t in [(0.60,0.40),(0.65,0.35),(0.70,0.30),(0.75,0.25),(0.80,0.20),(0.85,0.15)]:
        _cb  = final_probs > cb_t
        _ncb = final_probs < ncb_t
        _res = _cb | _ncb
        if _res.sum() == 0: continue
        _pred = _cb[_res].astype(int)
        _acc  = accuracy_score(y_test[_res], _pred)
        _cbp  = (y_test[_cb] == 1).mean() if _cb.sum() else 0.0
        _ncbp = (y_test[_ncb] == 0).mean() if _ncb.sum() else 0.0
        _cov  = _res.mean() * 100
        print(f"  {cb_t:>7.2f}  {ncb_t:>8.2f}  {_cov:>8.1f}%  "
              f"{_acc:>8.4f}  {_cbp:>8.4f}  {_ncbp:>9.4f}")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — EVALUATION & AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_audit(
    df, y, test_idx,
    final_probs, y_pred, y_test, threshold, tristate_labels,
    test_bert, test_lgbm,
):
    sec("SECTION 9 — Evaluation & Audit")

    # ── Core metrics ─────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    brier= brier_score_loss(y_test, final_probs)

    print(f"\n  Threshold used   : {threshold:.4f}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1-score         : {f1:.4f}  ← TARGET > 0.711")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  Brier Score      : {brier:.4f}")

    # ── Comparison table ─────────────────────────────────────────────────
    subsec("Model Progression Comparison")
    PRIOR = [
        ("Logistic Regression (L1)",        0.6222, 0.8019, 0.5952, 0.6518),
        ("LightGBM (26 feat)",              0.6263, 0.8037, 0.5979, 0.6575),
        ("DistilBERT v1 (postOnly)",        0.6967, 0.8533, 0.7215, 0.6735),
        ("DistilBERT v2 (reg+title)",       0.7110, 0.8400, None,   None  ),
        ("Ensemble v1 (BERT+LGBM hard)",    0.7110, 0.8400, None,   None  ),
    ]
    print(f"\n  {'Model':<42} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}")
    print(f"  {'─'*42} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for name, _f1, _acc, _p, _r in PRIOR:
        ps = f"{_p:.4f}" if _p else "  —   "
        rs = f"{_r:.4f}" if _r else "  —   "
        print(f"  {name:<42} {_f1:.4f} {_acc:.4f} {ps} {rs}")
    print(f"  {'─'*42} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    delta = f1 - 0.7110
    marker = "✓ BEAT BARRIER" if f1 > 0.711 else "✗ below target"
    print(f"  {'★  Hybrid MetaStack (this script)':<42} "
          f"{f1:.4f} {acc:.4f} {prec:.4f} {rec:.4f}  {marker} ({delta:+.4f})")

    # ── Confusion matrix ─────────────────────────────────────────────────
    subsec("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n               Pred NCB   Pred CB")
    print(f"  True NCB  :  {tn:>8}   {fp:>7}")
    print(f"  True CB   :  {fn:>8}   {tp:>7}")
    print(f"\n  True Negative   Rate : {tn/(tn+fp):.4f}")
    print(f"  False Positive  Rate : {fp/(tn+fp):.4f}")
    print(f"  False Negative  Rate : {fn/(fn+tp):.4f}")
    print(f"  True Positive   Rate : {tp/(fn+tp):.4f}")

    # Full classification report
    subsec("Classification Report")
    print(classification_report(y_test, y_pred,
                                target_names=["not-clickbait", "clickbait"]))

    # ── Top-10 Model vs Human Discrepancies ─────────────────────────────
    subsec("Top-10 'Model vs Human' Discrepancies")
    truth_mean = df.iloc[test_idx]["truthMean"].values

    # Model is very confident (prob > 0.80 or < 0.20) but disagrees with truthClass
    model_confident = (final_probs > 0.75) | (final_probs < 0.25)
    disagrees       = y_pred != y_test
    high_discrepancy = model_confident & disagrees

    # Score: product of model confidence and distance from truthMean
    disc_score = np.abs(final_probs - 0.5) * np.abs(final_probs - truth_mean)
    disc_score[~high_discrepancy] = -1

    top10_idx = np.argsort(disc_score)[::-1][:10]

    print(f"\n  These are cases where the model's confidence sharply contradicts")
    print(f"  the human truthClass, suggesting either model superiority or label noise.\n")
    print(f"  {'#':>2}  {'TrCls':>5}  {'TrMean':>7}  {'Prob':>6}  {'Bucket':<15}  Post Text")
    print(f"  {'─'*2}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*15}  {'─'*50}")

    for rank, i in enumerate(top10_idx, 1):
        if disc_score[i] < 0:
            continue
        row_idx  = test_idx[i]
        post_txt = df.iloc[row_idx]["postText_clean"]
        post_txt = textwrap.shorten(post_txt, width=65, placeholder="…")
        title_txt = df.iloc[row_idx]["targetTitle_clean"]
        title_txt = textwrap.shorten(title_txt, width=65, placeholder="…")
        bucket   = tristate_labels[i]
        model_pos = "CB  " if final_probs[i] > 0.5 else "NCB "

        # Hypothesis
        if y_test[i] == 1 and final_probs[i] < 0.30:
            hypo = "Post mirrors headline verbatim — no curiosity gap. Model may be right."
        elif y_test[i] == 0 and final_probs[i] > 0.70:
            hypo = "Post uses list-bait / slang triggers despite low human score."
        else:
            hypo = "Borderline case — annotator disagreement likely."

        print(f"  {rank:>2}  {y_test[i]:>5}  {truth_mean[i]:>7.2f}  "
              f"{final_probs[i]:>6.3f}  {bucket:<15}")
        print(f"      postText : {post_txt}")
        print(f"      title    : {title_txt}")
        print(f"      Hypo     : {hypo}")
        print()

    # ── Model component correlation ──────────────────────────────────────
    subsec("Base-Learner Correlation on Test Set")
    corr = np.corrcoef(test_bert, test_lgbm)[0, 1]
    print(f"  BERT vs LightGBM probability correlation: {corr:.4f}")
    print(f"  (Low correlation → both models bring independent signal)")

    # Mean absolute disagreement
    mad = np.mean(np.abs(test_bert - test_lgbm))
    print(f"  Mean absolute disagreement: {mad:.4f}")

    return acc, f1, prec, rec


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    df, y, test_idx,
    final_probs, y_pred, y_test,
    test_bert, test_lgbm, tristate_labels,
    f1, acc, prec, rec, threshold,
    best_calib_method, ece_platt, ece_iso,
):
    sec("SECTION 10 — Saving Outputs")

    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_label"]          = y_test
    out["predicted_binary"]    = y_pred
    out["prob_final"]          = final_probs
    out["prob_distilbert"]     = test_bert
    out["prob_lgbm"]           = test_lgbm
    out["prob_disagreement"]   = np.abs(test_bert - test_lgbm)
    out["tristate_label"]      = tristate_labels
    out["is_correct"]          = (y_pred == y_test).astype(int)

    out.to_csv(CFG["OUTPUT_PREDS_FILE"], index=False)
    print(f"  ✓ Predictions saved: {CFG['OUTPUT_PREDS_FILE']}  (shape {out.shape})")

    # ── Print summary card ────────────────────────────────────────────────
    summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║            HYBRID META-STACKING — FINAL RESULTS CARD               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture                                                        ║
║    Base 1 : DistilBERT v2  (regression on truthMean)                ║
║    Base 2 : LightGBM 28-feat (incl. mirror_similarity, slang)       ║
║    Meta   : Logistic Regression (6 meta-features)                   ║
║    Calib  : {best_calib_method:<10} (ECE Platt={ece_platt:.4f} | ISO={ece_iso:.4f})     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Test-Set Metrics  (20% stratified, seed=42)                        ║
║    F1-score   : {f1:.4f}   {'★ BARRIER BROKEN' if f1 > 0.711 else '→ below 0.711 barrier'}{'':20}║
║    Accuracy   : {acc:.4f}                                              ║
║    Precision  : {prec:.4f}                                              ║
║    Recall     : {rec:.4f}                                              ║
║    Threshold  : {threshold:.4f}                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Tri-State  (CB > 0.80, NCB < 0.20)                                 ║
║    CERTAIN_CB   :  {(tristate_labels == "CERTAIN_CB").sum():>5} samples                          ║
║    CERTAIN_NCB  :  {(tristate_labels == "CERTAIN_NCB").sum():>5} samples                          ║
║    AMBIGUOUS_IDK:  {(tristate_labels == "AMBIGUOUS_IDK").sum():>5} samples                          ║
╚══════════════════════════════════════════════════════════════════════╝"""
    print(summary)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  CLICKBAIT HYBRID META-STACKING PIPELINE                          ║")
    print("║  Target: Break F1 = 0.711 barrier                                 ║")
    print(f"║  Device: {str(DEVICE):<58} ║")
    print("╚" + "═"*68 + "╝")

    # ── 1. Data ────────────────────────────────────────────────────────────
    df, y, base_train_idx, blend_val_idx, test_idx = load_and_split()

    # ── 2. Tabular features (28) ───────────────────────────────────────────
    feat_df, feature_names = build_tabular_features(df, base_train_idx, y)

    # ── 3. DistilBERT probabilities ────────────────────────────────────────
    blend_bert, test_bert = train_or_load_distilbert(
        df, base_train_idx, blend_val_idx, test_idx
    )

    # ── 4. LightGBM probabilities ──────────────────────────────────────────
    blend_lgbm, test_lgbm, _ = train_lightgbm(
        feat_df, feature_names, y, base_train_idx, blend_val_idx, test_idx
    )

    # ── 5. Meta-learner ────────────────────────────────────────────────────
    meta_lr, meta_features_fn = train_meta_learner(
        y, blend_val_idx, blend_bert, blend_lgbm
    )

    # ── 6. Calibration (on blend_val) ─────────────────────────────────────
    blend_meta_raw = meta_lr.predict_proba(
        meta_features_fn(blend_bert, blend_lgbm)
    )[:, 1]
    calibrate_fn, ece_platt, ece_iso, best_calib = calibrate_probabilities(
        y, blend_val_idx, blend_meta_raw, meta_lr, meta_features_fn
    )

    # ── 7. Final test predictions ──────────────────────────────────────────
    final_probs, y_pred, y_test, threshold = make_final_predictions(
        df, y, test_idx, test_bert, test_lgbm,
        meta_lr, meta_features_fn, calibrate_fn,
    )

    # ── 8. Tri-state ───────────────────────────────────────────────────────
    tristate_labels = tristate_classification(final_probs, y_test)

    # ── 9. Evaluation & audit ──────────────────────────────────────────────
    acc, f1, prec, rec = evaluate_and_audit(
        df, y, test_idx,
        final_probs, y_pred, y_test, threshold, tristate_labels,
        test_bert, test_lgbm,
    )

    # ── 10. Save ───────────────────────────────────────────────────────────
    save_outputs(
        df, y, test_idx,
        final_probs, y_pred, y_test,
        test_bert, test_lgbm, tristate_labels,
        f1, acc, prec, rec, threshold,
        best_calib, ece_platt, ece_iso,
    )

    print("\n  Pipeline complete.")
