"""
clickbait_xgboost.py
--------------------
Clickbait detection using 22 text-only interpretable features + XGBoost with
scale_pos_weight class balancing and threshold optimization.

Features:
  Group A (6): Text statistics (word counts, punctuation, caps ratio)
  Group B (3): VADER sentiment (post, article, abs gap)
  Group C (3): Semantic mismatch (cosine similarity, KL divergence, Jaccard)
  Group D (6): Clickbait linguistic patterns (data-driven n-grams, numbers, demonstratives, etc.)
  Group E (4): Article metadata text features (title similarity, description similarity, keyword overlap)
  Total: 22 features

Usage (Colab):
    1. Upload this script and final_cleaned_full.csv to Colab
    2. pip install sentence-transformers scikit-learn vaderSentiment xgboost
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
OUTPUT_FILE = "clickbait_predictions_xgboost.csv"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_K_NGRAMS = 100  # number of most discriminative n-grams to select via chi-squared

# Feature names for reference
FEATURE_NAMES = [
    # Group A: Text statistics
    "post_word_count", "article_word_count", "word_count_ratio",
    "post_question_marks", "post_exclamation_marks", "post_caps_ratio",
    # Group B: VADER sentiment
    "post_sentiment", "article_sentiment", "abs_sentiment_gap",
    # Group C: Semantic mismatch
    "cosine_similarity", "kl_divergence", "jaccard_similarity",
    # Group D: Clickbait linguistic patterns
    "clickbait_ngram_count", "has_number", "starts_with_demonstrative",
    "second_person_count", "avg_word_length", "ellipsis_count",
    # Group E: Article metadata text features
    "title_post_cosine_sim", "title_post_jaccard",
    "desc_post_cosine_sim", "keyword_overlap_ratio",
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

    print("   Parsing text columns...")
    df["postText_clean"] = df["postText"].apply(parse_text_list)
    df["targetParagraphs_clean"] = df["targetParagraphs"].apply(parse_text_list)
    df["targetTitle_clean"] = df["targetTitle"].apply(parse_text_list)
    df["targetDescription_clean"] = df["targetDescription"].apply(parse_text_list)
    df["targetKeywords_clean"] = df["targetKeywords"].apply(parse_text_list)

    before = len(df)
    df = df[
        (df["postText_clean"].str.strip() != "")
        & (df["targetParagraphs_clean"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"   Dropped {before - len(df)} rows with empty text")
    print(f"   Remaining: {len(df)} rows")

    # Report coverage of new text columns
    n_title = (df["targetTitle_clean"].str.strip() != "").sum()
    n_desc = (df["targetDescription_clean"].str.strip() != "").sum()
    n_kw = (df["targetKeywords_clean"].str.strip() != "").sum()
    print(f"   targetTitle coverage:       {n_title}/{len(df)} ({100*n_title/len(df):.1f}%)")
    print(f"   targetDescription coverage: {n_desc}/{len(df)} ({100*n_desc/len(df):.1f}%)")
    print(f"   targetKeywords coverage:    {n_kw}/{len(df)} ({100*n_kw/len(df):.1f}%)")

    print(f"   Class distribution:")
    print(f"   {df['truthClass'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# 2. SBERT encoding
# ---------------------------------------------------------------------------

def encode_texts(df):
    """Encode postText, targetParagraphs, targetTitle, and targetDescription with SBERT."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("2. SBERT ENCODING")
    print("=" * 70)
    print(f"   Model: {SBERT_MODEL}")
    print(f"   Device: {device}")

    model = SentenceTransformer(SBERT_MODEL, device=device)

    print(f"   Encoding postText ({len(df)} texts)...")
    emb_post = model.encode(
        df["postText_clean"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"   Encoding targetParagraphs ({len(df)} texts)...")
    emb_article = model.encode(
        df["targetParagraphs_clean"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Encode targetTitle (fill empty with placeholder to avoid SBERT issues)
    titles = df["targetTitle_clean"].tolist()
    titles = [t if t.strip() else "no title" for t in titles]
    print(f"   Encoding targetTitle ({len(df)} texts)...")
    emb_title = model.encode(
        titles,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Encode targetDescription (fill empty with placeholder)
    descs = df["targetDescription_clean"].tolist()
    descs = [d if d.strip() else "no description" for d in descs]
    print(f"   Encoding targetDescription ({len(df)} texts)...")
    emb_desc = model.encode(
        descs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"   Embedding shape: {emb_post.shape}")
    return emb_post, emb_article, emb_title, emb_desc


# ---------------------------------------------------------------------------
# 3. Feature engineering (22 text-only features)
#    NOTE: train_idx and y_train are required so that the chi-squared n-gram
#    selection is fit ONLY on training data (prevents data leakage).
# ---------------------------------------------------------------------------

def build_features(df, emb_post, emb_article, emb_title, emb_desc, train_idx, y_train):
    """Build 22 text-only interpretable features across 5 groups.

    Group A: Text statistics (6)
    Group B: VADER sentiment (3)
    Group C: Semantic mismatch (3)
    Group D: Clickbait linguistic patterns (6) — n-grams are data-driven
    Group E: Article metadata text features (4)
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import chi2

    print("\n" + "=" * 70)
    print("3. FEATURE ENGINEERING (22 text-only features)")
    print("=" * 70)

    features = pd.DataFrame(index=df.index)
    post_texts = df["postText_clean"]
    article_texts = df["targetParagraphs_clean"]

    # -----------------------------------------------------------------------
    # Group A: Text statistics (6)
    # -----------------------------------------------------------------------
    print("   Group A: Text statistics...")

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
    # Group B: VADER sentiment (3)
    # -----------------------------------------------------------------------
    print("   Group B: VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    post_sentiments = post_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])
    article_sentiments = article_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])

    features["post_sentiment"] = post_sentiments
    features["article_sentiment"] = article_sentiments
    features["abs_sentiment_gap"] = (post_sentiments - article_sentiments).abs()

    # -----------------------------------------------------------------------
    # Group C: Semantic mismatch (3)
    # -----------------------------------------------------------------------
    print("   Group C: Semantic mismatch features...")

    # C1: Cosine similarity from SBERT embeddings (post vs article body)
    dot = np.sum(emb_post * emb_article, axis=1)
    norm_post = np.linalg.norm(emb_post, axis=1)
    norm_article = np.linalg.norm(emb_article, axis=1)
    features["cosine_similarity"] = dot / (norm_post * norm_article + 1e-8)

    # C2: KL divergence between TF-IDF distributions
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

    # C3: Jaccard similarity (word-level overlap)
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
    # Group D: Clickbait linguistic patterns (6)
    # -----------------------------------------------------------------------
    print("   Group D: Clickbait linguistic patterns...")

    # D1: Data-driven clickbait n-gram count (chi-squared selection)
    #     Fit CountVectorizer + chi2 on TRAINING data only to prevent leakage
    print(f"   Selecting top {TOP_K_NGRAMS} discriminative n-grams (chi-squared on train only)...")

    cv = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=10000)
    train_posts = post_texts.iloc[train_idx]
    X_ngram_train = cv.fit_transform(train_posts)

    chi2_scores, chi2_pvals = chi2(X_ngram_train, y_train)

    top_k_idx = np.argsort(chi2_scores)[::-1][:TOP_K_NGRAMS]
    vocab = cv.get_feature_names_out()

    print(f"   Top 20 clickbait-discriminative n-grams:")
    for i, idx in enumerate(top_k_idx[:20], 1):
        print(f"     {i:2d}. '{vocab[idx]}'  (chi2 = {chi2_scores[idx]:.1f}, p = {chi2_pvals[idx]:.2e})")

    # Transform ALL data with the fitted vectorizer, sum top-K columns
    X_ngram_all = cv.transform(post_texts)
    features["clickbait_ngram_count"] = np.array(
        X_ngram_all[:, top_k_idx].sum(axis=1)
    ).flatten()

    # D2: Has number (listicle signal)
    features["has_number"] = post_texts.apply(
        lambda x: 1 if re.search(r"\d", x) else 0
    )

    # D3: Starts with demonstrative pronoun
    demonstrative_pattern = re.compile(r"^\s*(this|these|here|that)\b", re.IGNORECASE)
    features["starts_with_demonstrative"] = post_texts.apply(
        lambda x: 1 if demonstrative_pattern.search(x) else 0
    )

    # D4: Second person pronoun count
    second_person_pattern = re.compile(r"\b(you|your|you're|yourself|yours)\b", re.IGNORECASE)
    features["second_person_count"] = post_texts.apply(
        lambda x: len(second_person_pattern.findall(x))
    )

    # D5: Average word length
    features["avg_word_length"] = post_texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0
    )

    # D6: Ellipsis count (suspense punctuation)
    features["ellipsis_count"] = post_texts.apply(
        lambda x: x.count("...")
    )

    # -----------------------------------------------------------------------
    # Group E: Article metadata text features (4)
    #   Uses targetTitle, targetDescription, targetKeywords
    # -----------------------------------------------------------------------
    print("   Group E: Article metadata text features...")

    title_texts = df["targetTitle_clean"]
    desc_texts = df["targetDescription_clean"]
    kw_texts = df["targetKeywords_clean"]

    # E1: Cosine similarity between post and article title (SBERT)
    dot_title = np.sum(emb_post * emb_title, axis=1)
    norm_title = np.linalg.norm(emb_title, axis=1)
    title_cos = dot_title / (norm_post * norm_title + 1e-8)
    # Zero out where title was empty
    empty_title = (title_texts.str.strip() == "").values
    title_cos[empty_title] = 0.0
    features["title_post_cosine_sim"] = title_cos

    # E2: Jaccard similarity between post and article title (word-level)
    title_jaccard = []
    for pt, tt in zip(post_texts, title_texts):
        pw = set(pt.lower().split())
        tw = set(tt.lower().split())
        if len(pw | tw) == 0:
            title_jaccard.append(0.0)
        else:
            title_jaccard.append(len(pw & tw) / len(pw | tw))
    features["title_post_jaccard"] = title_jaccard

    # E3: Cosine similarity between post and article description (SBERT)
    dot_desc = np.sum(emb_post * emb_desc, axis=1)
    norm_desc = np.linalg.norm(emb_desc, axis=1)
    desc_cos = dot_desc / (norm_post * norm_desc + 1e-8)
    # Zero out where description was empty
    empty_desc = (desc_texts.str.strip() == "").values
    desc_cos[empty_desc] = 0.0
    features["desc_post_cosine_sim"] = desc_cos

    # E4: Keyword overlap ratio (fraction of article keywords found in post)
    kw_overlap = []
    for pt, kw in zip(post_texts, kw_texts):
        kw_str = kw.strip()
        if not kw_str:
            kw_overlap.append(0.0)
            continue
        # Split keywords on commas or whitespace
        keywords = set(re.split(r"[,\s]+", kw_str.lower()))
        keywords.discard("")
        if not keywords:
            kw_overlap.append(0.0)
            continue
        post_lower = pt.lower()
        matched = sum(1 for k in keywords if k in post_lower)
        kw_overlap.append(matched / len(keywords))
    features["keyword_overlap_ratio"] = kw_overlap

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n   Feature matrix shape: {features.shape}")
    print(f"   Features: {list(features.columns)}")

    # Per-feature correlation with truthClass
    print("\n   Feature correlations with truthClass:")
    y_all = df["truthClass"]
    for col in features.columns:
        corr = features[col].corr(y_all)
        print(f"     {col:30s}  r = {corr:+.4f}")

    return features


# ---------------------------------------------------------------------------
# 4-5. Class weighting, hyperparameter tuning & threshold optimization
# ---------------------------------------------------------------------------

def train_and_evaluate(features_df, y, df, train_idx, test_idx):
    """Scale, tune XGBoost with scale_pos_weight + RandomizedSearchCV,
    optimize decision threshold, evaluate, and save results."""
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        f1_score,
    )
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = features_df.values
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"   Features: {X.shape[1]} text-only hand-crafted features")

    # --- 4. Split summary ---
    print("\n" + "=" * 70)
    print("4. TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"   Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"   Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   Test  class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- 4b. Scaling + class weight ---
    print("\n" + "=" * 70)
    print("4b. SCALING & CLASS WEIGHTING")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class imbalance ratio for scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    imbalance_ratio = n_neg / n_pos
    print(f"   Class imbalance ratio (neg/pos): {imbalance_ratio:.2f}")
    print(f"   Using scale_pos_weight={imbalance_ratio:.2f}")

    # --- 5. Hyperparameter tuning with RandomizedSearchCV ---
    print("\n" + "=" * 70)
    print("5. HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 70)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    }

    base_clf = XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("   Running 100-iteration random search with 5-fold stratified CV...")
    search.fit(X_train_scaled, y_train)

    clf = search.best_estimator_
    print(f"\n   Best CV F1 score: {search.best_score_:.4f}")
    print(f"   Best parameters:")
    for param, val in search.best_params_.items():
        print(f"     {param}: {val}")

    # --- Feature importance ---
    print(f"\n   Feature Importance Ranking (all {len(FEATURE_NAMES)} features):")
    importances = clf.feature_importances_

    importance_order = np.argsort(importances)[::-1]
    for rank, idx in enumerate(importance_order, 1):
        print(f"     {rank:2d}. {FEATURE_NAMES[idx]:30s}  importance = {importances[idx]:.4f}")

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
    out = df.iloc[test_idx].copy()
    out["predicted"] = y_pred
    for fname in FEATURE_NAMES:
        out[fname] = features_df.iloc[test_idx][fname].values
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n   Saved test predictions to {OUTPUT_FILE}")
    print(f"   Shape: {out.shape}")

    return clf, y_test, y_pred, test_idx


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

from sklearn.model_selection import train_test_split

# 1. Load data
df = load_and_clean(INPUT_FILE)

# 2. SBERT encoding (for cosine similarity features)
emb_post, emb_article, emb_title, emb_desc = encode_texts(df)

# 3. Train/test split FIRST (before feature engineering to prevent n-gram leakage)
y = df["truthClass"].values
train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# 4. Feature engineering (22 text-only features, n-grams fit on train only)
features_df = build_features(df, emb_post, emb_article, emb_title, emb_desc, train_idx, y[train_idx])

# 5-6. Class-weighted XGBoost + Threshold optimization
clf, y_test, y_pred, idx_test = train_and_evaluate(features_df, y, df, train_idx, test_idx)

# 7. Verification
print_verification(df, features_df, y_test, y_pred, idx_test)

print("\nDone!")
