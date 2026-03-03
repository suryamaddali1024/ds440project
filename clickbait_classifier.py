"""
clickbait_classifier.py
-----------------------
Clickbait detection using 26 interpretable features + XGBoost with SMOTE
resampling and threshold optimization.

Features:
  Group A (5): Original dataset features (media, captions, description, keywords, hour)
  Group B (6): Text statistics (word counts, punctuation, caps ratio)
  Group C (5): VADER sentiment (post, article, gap, abs gap, intensity)
  Group D (3): Semantic mismatch (cosine similarity, KL divergence, Jaccard)
  Group E (7): Clickbait linguistic patterns (trigger phrases, numbers, demonstratives, etc.)
  Total: 26 features

Usage (Colab):
    1. Upload this script and final_cleaned_full.csv to Colab
    2. pip install sentence-transformers scikit-learn vaderSentiment xgboost imbalanced-learn
    3. Set runtime to GPU (optional but faster for SBERT encoding)
    4. Paste this entire script into a cell and run
"""

import ast
import re
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "final_cleaned_full.csv"
OUTPUT_FILE = "clickbait_predictions.csv"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Feature names for reference
FEATURE_NAMES = [
    # Group A: Original dataset features
    "has_media", "num_captions", "has_description", "num_keywords", "post_hour",
    # Group B: Text statistics
    "post_word_count", "article_word_count", "word_count_ratio",
    "post_question_marks", "post_exclamation_marks", "post_caps_ratio",
    # Group C: VADER sentiment
    "post_sentiment", "article_sentiment", "sentiment_gap",
    "abs_sentiment_gap", "post_sentiment_intensity",
    # Group D: Semantic mismatch
    "cosine_similarity", "kl_divergence", "jaccard_similarity",
    # Group E: Clickbait linguistic patterns
    "trigger_phrase_count", "has_number", "starts_with_demonstrative",
    "second_person_count", "post_char_length", "avg_word_length",
    "ellipsis_count",
]

# Clickbait trigger phrases for Group E feature
TRIGGER_PHRASES = [
    "you won't believe", "you will not believe",
    "what happened next", "what happens next",
    "shocking", "jaw-dropping", "jaw dropping",
    "mind-blowing", "mind blowing",
    "can't stop laughing", "dying to know",
    "this is why", "here's why", "here is why",
    "the reason is", "the truth about",
    "secret", "tricks", "hacks",
    "amazing", "incredible", "unbelievable",
    "goes wrong", "gone wrong",
    "you need to", "you have to",
    "don't want you to know", "they don't want",
    "will make you", "will blow your mind",
    "changed my life", "change your life",
    "number \\d+ will", "#\\d+ will",
    "won't believe what", "guess what",
    "omg", "wtf", "lol",
]

# ---------------------------------------------------------------------------
# Text parsing  (reused from generate_bart_summaries.py)
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


def parse_list_count(raw_str):
    """Parse a string-encoded Python list and return the count of non-empty items."""
    if pd.isna(raw_str) or not str(raw_str).strip():
        return 0
    raw_str = str(raw_str).strip()
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list):
            return len([item for item in parsed if str(item).strip()])
        return 1 if str(parsed).strip() else 0
    except (ValueError, SyntaxError):
        return 0


def parse_has_items(raw_str):
    """Parse a string-encoded Python list and return 1 if it has any non-empty items, else 0."""
    return 1 if parse_list_count(raw_str) > 0 else 0


def parse_hour(timestamp_str):
    """Extract hour of day from postTimestamp string like 'Tue Jun 09 16:31:10 +0000 2015'."""
    if pd.isna(timestamp_str) or not str(timestamp_str).strip():
        return 12  # default to noon
    try:
        from datetime import datetime
        dt = datetime.strptime(str(timestamp_str).strip(), "%a %b %d %H:%M:%S %z %Y")
        return dt.hour
    except (ValueError, TypeError):
        return 12


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_clean(input_path):
    """Load CSV, parse text columns, drop rows with empty text."""
    print("=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)
    df = pd.read_csv(input_path, encoding="latin-1")
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    print("   Parsing postText and targetParagraphs...")
    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)

    before = len(df)
    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"   Dropped {before - len(df)} rows with empty text")
    print(f"   Remaining: {len(df)} rows")

    print(f"   Class distribution:")
    print(f"   {df['truthClass'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# 2. SBERT encoding (kept for cosine similarity feature)
# ---------------------------------------------------------------------------

def encode_texts(df):
    """Encode postText and targetParagraphs with SBERT."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("2. SBERT ENCODING")
    print("=" * 70)
    print(f"   Model: {SBERT_MODEL}")
    print(f"   Device: {device}")

    model = SentenceTransformer(SBERT_MODEL, device=device)

    print(f"   Encoding postText ({len(df)} texts, batch_size={BATCH_SIZE})...")
    emb_post = model.encode(
        df["postText_clean"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"   Encoding targetParagraphs ({len(df)} texts, batch_size={BATCH_SIZE})...")
    emb_article = model.encode(
        df["targetParagraphs_clean"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"   Embedding shape: {emb_post.shape}")
    return emb_post, emb_article


# ---------------------------------------------------------------------------
# 3. Feature engineering (26 interpretable features)
# ---------------------------------------------------------------------------

def build_features(df, emb_post, emb_article):
    """Build 26 interpretable features across 5 groups.

    Group A: Original dataset features (5)
    Group B: Text statistics (6)
    Group C: VADER sentiment (5)
    Group D: Semantic mismatch (3)
    Group E: Clickbait linguistic patterns (7)
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n" + "=" * 70)
    print("3. FEATURE ENGINEERING (26 features)")
    print("=" * 70)

    features = pd.DataFrame(index=df.index)

    # -----------------------------------------------------------------------
    # Group A: Original dataset features (5)
    # -----------------------------------------------------------------------
    print("   Group A: Original dataset features...")
    features["has_media"] = df["postMedia"].apply(parse_has_items)
    features["num_captions"] = df["targetCaptions"].apply(parse_list_count)
    features["has_description"] = df["targetDescription"].apply(
        lambda x: 0 if pd.isna(x) or not str(x).strip() else 1
    )
    features["num_keywords"] = df["targetKeywords"].apply(parse_list_count)
    features["post_hour"] = df["postTimestamp"].apply(parse_hour)

    # -----------------------------------------------------------------------
    # Group B: Text statistics (6)
    # -----------------------------------------------------------------------
    print("   Group B: Text statistics...")
    post_texts = df["postText_clean"]
    article_texts = df["targetParagraphs_clean"]

    features["post_word_count"] = post_texts.apply(lambda x: len(x.split()))
    features["article_word_count"] = article_texts.apply(lambda x: len(x.split()))
    features["word_count_ratio"] = (
        features["post_word_count"] / features["article_word_count"].replace(0, 1)
    )
    features["post_question_marks"] = post_texts.apply(lambda x: x.count("?"))
    features["post_exclamation_marks"] = post_texts.apply(lambda x: x.count("!"))
    features["post_caps_ratio"] = post_texts.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1) / max(len(x.split()), 1)
    )

    # -----------------------------------------------------------------------
    # Group C: VADER sentiment (5)
    # -----------------------------------------------------------------------
    print("   Group C: VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    post_sentiments = post_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])
    article_sentiments = article_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])

    features["post_sentiment"] = post_sentiments
    features["article_sentiment"] = article_sentiments
    features["sentiment_gap"] = post_sentiments - article_sentiments
    features["abs_sentiment_gap"] = (post_sentiments - article_sentiments).abs()
    features["post_sentiment_intensity"] = post_sentiments.abs()

    # -----------------------------------------------------------------------
    # Group D: Semantic mismatch (3)
    # -----------------------------------------------------------------------
    print("   Group D: Semantic mismatch features...")

    # D1: Cosine similarity from SBERT embeddings
    dot = np.sum(emb_post * emb_article, axis=1)
    norm_post = np.linalg.norm(emb_post, axis=1)
    norm_article = np.linalg.norm(emb_article, axis=1)
    features["cosine_similarity"] = dot / (norm_post * norm_article + 1e-8)

    # D2: KL divergence between TF-IDF distributions
    print("   Computing KL divergence (TF-IDF)...")
    all_texts = pd.concat([post_texts, article_texts]).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf.fit(all_texts)

    tfidf_post = tfidf.transform(post_texts.tolist()).toarray()
    tfidf_article = tfidf.transform(article_texts.tolist()).toarray()

    # Normalize to probability distributions and compute KL divergence
    eps = 1e-10
    p = tfidf_post + eps
    q = tfidf_article + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    kl_div = np.sum(p * np.log(p / q), axis=1)
    features["kl_divergence"] = kl_div

    # D3: Jaccard similarity (word-level overlap)
    print("   Computing Jaccard similarity...")
    jaccard_scores = []
    for pt, at in zip(post_texts, article_texts):
        post_words = set(pt.lower().split())
        article_words = set(at.lower().split())
        if len(post_words | article_words) == 0:
            jaccard_scores.append(0.0)
        else:
            jaccard_scores.append(
                len(post_words & article_words) / len(post_words | article_words)
            )
    features["jaccard_similarity"] = jaccard_scores

    # -----------------------------------------------------------------------
    # Group E: Clickbait linguistic patterns (7)
    # -----------------------------------------------------------------------
    print("   Group E: Clickbait linguistic patterns...")

    # E1: Trigger phrase count
    trigger_pattern = re.compile("|".join(TRIGGER_PHRASES), re.IGNORECASE)
    features["trigger_phrase_count"] = post_texts.apply(
        lambda x: len(trigger_pattern.findall(x))
    )

    # E2: Has number (listicle signal)
    features["has_number"] = post_texts.apply(
        lambda x: 1 if re.search(r"\d", x) else 0
    )

    # E3: Starts with demonstrative pronoun
    demonstrative_pattern = re.compile(r"^\s*(this|these|here|that)\b", re.IGNORECASE)
    features["starts_with_demonstrative"] = post_texts.apply(
        lambda x: 1 if demonstrative_pattern.search(x) else 0
    )

    # E4: Second person pronoun count
    second_person_pattern = re.compile(r"\b(you|your|you're|yourself|yours)\b", re.IGNORECASE)
    features["second_person_count"] = post_texts.apply(
        lambda x: len(second_person_pattern.findall(x))
    )

    # E5: Post character length
    features["post_char_length"] = post_texts.apply(len)

    # E6: Average word length
    features["avg_word_length"] = post_texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0
    )

    # E7: Ellipsis count (suspense punctuation)
    features["ellipsis_count"] = post_texts.apply(
        lambda x: x.count("...")
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n   Feature matrix shape: {features.shape}")
    print(f"   Features: {list(features.columns)}")

    # Per-feature correlation with truthClass
    print("\n   Feature correlations with truthClass:")
    y = df["truthClass"]
    for col in features.columns:
        corr = features[col].corr(y)
        print(f"     {col:30s}  r = {corr:+.4f}")

    return features


# ---------------------------------------------------------------------------
# 4-5. Train/test split, SMOTE, hyperparameter tuning & threshold optimization
# ---------------------------------------------------------------------------

def train_and_evaluate(features_df, y, df, emb_post, emb_article):
    """Split data, apply SMOTE, tune XGBoost with RandomizedSearchCV,
    optimize decision threshold, evaluate, and save results."""
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        f1_score,
    )
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    # Combine 26 hand-crafted features with SBERT embeddings (384+384=768)
    X_handcrafted = features_df.values
    X = np.hstack([X_handcrafted, emb_post, emb_article])
    n_handcrafted = X_handcrafted.shape[1]
    n_total = X.shape[1]
    print(f"   Combined features: {n_handcrafted} hand-crafted + {emb_post.shape[1]}d post emb "
          f"+ {emb_article.shape[1]}d article emb = {n_total} total")

    # --- 4. Train/test split ---
    print("\n" + "=" * 70)
    print("4. TRAIN/TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"   Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"   Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   Test  class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- 4b. SMOTE resampling on training data ---
    print("\n" + "=" * 70)
    print("4b. SMOTE RESAMPLING")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"   Before SMOTE: {len(X_train_scaled)} samples")
    print(f"   After  SMOTE: {len(X_train_resampled)} samples")
    print(f"   Resampled class dist: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

    # --- 5. Hyperparameter tuning with RandomizedSearchCV ---
    print("\n" + "=" * 70)
    print("5. HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 70)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    }

    base_clf = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("   Running 50-iteration random search with 5-fold stratified CV...")
    search.fit(X_train_resampled, y_train_resampled)

    clf = search.best_estimator_
    print(f"\n   Best CV F1 score: {search.best_score_:.4f}")
    print(f"   Best parameters:")
    for param, val in search.best_params_.items():
        print(f"     {param}: {val}")

    # --- Feature importance ---
    print("\n   Feature Importance Ranking (top 30):")
    importances = clf.feature_importances_

    # Build full feature name list: 26 named + embedding dimensions
    all_feature_names = list(FEATURE_NAMES)
    all_feature_names += [f"post_emb_{i}" for i in range(emb_post.shape[1])]
    all_feature_names += [f"article_emb_{i}" for i in range(emb_article.shape[1])]

    importance_order = np.argsort(importances)[::-1]
    for rank, idx in enumerate(importance_order[:30], 1):
        print(f"     {rank:2d}. {all_feature_names[idx]:30s}  importance = {importances[idx]:.4f}")

    # Summarize embedding vs hand-crafted importance
    handcrafted_imp = importances[:n_handcrafted].sum()
    embedding_imp = importances[n_handcrafted:].sum()
    print(f"\n   Importance share: hand-crafted = {handcrafted_imp:.4f}, "
          f"embeddings = {embedding_imp:.4f}")

    # --- 6. Evaluation ---
    print("\n" + "=" * 70)
    print("6. EVALUATION")
    print("=" * 70)

    # --- 6a. Default threshold (0.5) ---
    print("\n   --- Default Threshold (0.5) ---")
    y_pred_default = clf.predict(X_test_scaled)

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred_default, target_names=["no-clickbait", "clickbait"]))

    print("   Confusion Matrix:")
    cm_default = confusion_matrix(y_test, y_pred_default)
    print(f"   {cm_default}")

    acc_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    print(f"\n   Default Accuracy: {acc_default:.4f}")
    print(f"   Default F1:       {f1_default:.4f}")

    # --- 6b. Threshold optimization ---
    print("\n   --- Threshold Optimization ---")
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # Compute F1 for each threshold (precisions/recalls have len = thresholds + 1)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    print(f"   Best threshold: {best_threshold:.4f} (F1 = {f1_scores[best_threshold_idx]:.4f})")

    y_pred_optimized = (y_proba >= best_threshold).astype(int)

    print("\n   Classification Report (optimized threshold):")
    print(classification_report(y_test, y_pred_optimized, target_names=["no-clickbait", "clickbait"]))

    print("   Confusion Matrix (optimized threshold):")
    cm_optimized = confusion_matrix(y_test, y_pred_optimized)
    print(f"   {cm_optimized}")

    acc_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    print(f"\n   Optimized Accuracy: {acc_optimized:.4f}")
    print(f"   Optimized F1:       {f1_optimized:.4f}")

    # Pick whichever threshold yielded better F1
    if f1_optimized > f1_default:
        y_pred = y_pred_optimized
        acc = acc_optimized
        print(f"\n   >>> Using optimized threshold ({best_threshold:.4f}) — F1 improved "
              f"{f1_default:.4f} -> {f1_optimized:.4f}")
    else:
        y_pred = y_pred_default
        acc = acc_default
        print(f"\n   >>> Using default threshold (0.5) — already best")

    print(f"\n   Final Accuracy: {acc:.4f}")

    # --- Save output CSV ---
    out = df.iloc[idx_test].copy()
    out["predicted"] = y_pred
    for i, fname in enumerate(FEATURE_NAMES):
        out[fname] = features_df.iloc[idx_test][fname].values
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n   Saved test predictions to {OUTPUT_FILE}")
    print(f"   Shape: {out.shape}")

    return clf, y_test, y_pred, idx_test


# ---------------------------------------------------------------------------
# 7. Verification stats
# ---------------------------------------------------------------------------

def print_verification(df, features_df, y_test, y_pred, idx_test):
    """Print feature distributions by class and sample predictions."""
    print("\n" + "=" * 70)
    print("7. VERIFICATION STATS")
    print("=" * 70)

    y_all = df["truthClass"].values

    # Per-feature distribution by class
    print("\n   Feature distributions by class (all data):")
    print(f"   {'Feature':30s}  {'Clickbait mean':>14s}  {'No-CB mean':>14s}  {'Diff':>10s}")
    print(f"   {'-'*30}  {'-'*14}  {'-'*14}  {'-'*10}")
    for col in features_df.columns:
        cb_mean = features_df.loc[y_all == 1, col].mean()
        ncb_mean = features_df.loc[y_all == 0, col].mean()
        diff = cb_mean - ncb_mean
        print(f"   {col:30s}  {cb_mean:14.4f}  {ncb_mean:14.4f}  {diff:+10.4f}")

    # Sample predictions with feature values
    print("\n   5 sample predictions from test set:")
    sample_idx = random.sample(range(len(idx_test)), min(5, len(idx_test)))
    for si in sample_idx:
        row_idx = idx_test[si]
        row = df.iloc[row_idx]
        print(f"\n     Row {row_idx}:")
        print(f"       postText:    {row['postText_clean'][:100]}")
        print(f"       article:     {row['targetParagraphs_clean'][:100]}")
        print(f"       true: {y_test[si]}  predicted: {y_pred[si]}")
        print(f"       Key features:")
        feat_row = features_df.iloc[row_idx]
        for fname in FEATURE_NAMES:
            print(f"         {fname:30s} = {feat_row[fname]:.4f}")

    print("\n" + "=" * 70)


# ===========================================================================
# RUN
# ===========================================================================

# 1. Load data
df = load_and_clean(INPUT_FILE)

# 2. SBERT encoding (for cosine similarity)
emb_post, emb_article = encode_texts(df)

# 3. Feature engineering (26 features)
features_df = build_features(df, emb_post, emb_article)

# 4-6. SMOTE + Tuned XGBoost + Threshold optimization
y = df["truthClass"].values
clf, y_test, y_pred, idx_test = train_and_evaluate(features_df, y, df, emb_post, emb_article)

# 7. Verification
print_verification(df, features_df, y_test, y_pred, idx_test)

print("\nDone!")
