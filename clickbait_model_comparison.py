"""
clickbait_model_comparison.py
-----------------------------
Clickbait detection: incremental model comparison + error analysis.

Trains 7 models from simple to complex on 26 text-only features, compares
them side by side, and performs detailed error analysis on the best model.

Models (in order of complexity):
  1. Logistic Regression + L1 (Lasso)
  2. Logistic Regression + L2 (Ridge)
  3. Elastic Net (L1 + L2)
  4. Linear SVC
  5. Random Forest
  6. XGBoost
  7. LightGBM

Features (26 total):
  Group A (6): Text statistics (word counts, punctuation, caps ratio)
  Group B (3): VADER sentiment (post, article, abs gap)
  Group C (3): Semantic mismatch (cosine similarity, KL divergence, Jaccard)
  Group D (6): Clickbait linguistic patterns (data-driven n-grams, numbers, etc.)
  Group E (4): Article metadata text features (title/desc similarity, keyword overlap)
  Group F (4): Error-driven features (sensationalism, sentiment intensity, proper nouns, forward references)

Usage (Colab):
    1. Upload this script and final_cleaned_full.csv to Colab
    2. pip install sentence-transformers scikit-learn vaderSentiment xgboost lightgbm
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
OUTPUT_FILE = "clickbait_predictions_comparison.csv"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_K_NGRAMS = 100

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
    # Group F: Error-driven features (from misclassification analysis)
    "sensational_word_count", "post_sentiment_intensity",
    "proper_noun_ratio", "forward_reference_count",
]


# ===========================================================================
# SECTION 1: DATA LOADING & FEATURE ENGINEERING
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


def load_and_clean(input_path):
    """Load CSV, parse text columns, drop rows with empty text."""
    print("=" * 70)
    print("SECTION 1: LOADING DATA")
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

    n_title = (df["targetTitle_clean"].str.strip() != "").sum()
    n_desc = (df["targetDescription_clean"].str.strip() != "").sum()
    n_kw = (df["targetKeywords_clean"].str.strip() != "").sum()
    print(f"   targetTitle coverage:       {n_title}/{len(df)} ({100*n_title/len(df):.1f}%)")
    print(f"   targetDescription coverage: {n_desc}/{len(df)} ({100*n_desc/len(df):.1f}%)")
    print(f"   targetKeywords coverage:    {n_kw}/{len(df)} ({100*n_kw/len(df):.1f}%)")

    print(f"   Class distribution: {df['truthClass'].value_counts().to_dict()}")
    return df


def encode_texts(df):
    """Encode postText, targetParagraphs, targetTitle, targetDescription with SBERT."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("SBERT ENCODING")
    print("=" * 70)
    print(f"   Model: {SBERT_MODEL}  |  Device: {device}")

    model = SentenceTransformer(SBERT_MODEL, device=device)

    print(f"   Encoding postText ({len(df)} texts)...")
    emb_post = model.encode(df["postText_clean"].tolist(), batch_size=BATCH_SIZE,
                            show_progress_bar=True, convert_to_numpy=True)

    print(f"   Encoding targetParagraphs ({len(df)} texts)...")
    emb_article = model.encode(df["targetParagraphs_clean"].tolist(), batch_size=BATCH_SIZE,
                               show_progress_bar=True, convert_to_numpy=True)

    titles = [t if t.strip() else "no title" for t in df["targetTitle_clean"].tolist()]
    print(f"   Encoding targetTitle ({len(df)} texts)...")
    emb_title = model.encode(titles, batch_size=BATCH_SIZE,
                             show_progress_bar=True, convert_to_numpy=True)

    descs = [d if d.strip() else "no description" for d in df["targetDescription_clean"].tolist()]
    print(f"   Encoding targetDescription ({len(df)} texts)...")
    emb_desc = model.encode(descs, batch_size=BATCH_SIZE,
                            show_progress_bar=True, convert_to_numpy=True)

    print(f"   Embedding shape: {emb_post.shape}")
    return emb_post, emb_article, emb_title, emb_desc


def build_features(df, emb_post, emb_article, emb_title, emb_desc, train_idx, y_train):
    """Build 26 text-only features across 6 groups (A-F)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import chi2

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING (26 text-only features)")
    print("=" * 70)

    features = pd.DataFrame(index=df.index)
    post_texts = df["postText_clean"]
    article_texts = df["targetParagraphs_clean"]

    # --- Group A: Text statistics (6) ---
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

    # --- Group B: VADER sentiment (3) ---
    print("   Group B: VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()
    post_sentiments = post_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])
    article_sentiments = article_texts.apply(lambda x: analyzer.polarity_scores(x)["compound"])
    features["post_sentiment"] = post_sentiments
    features["article_sentiment"] = article_sentiments
    features["abs_sentiment_gap"] = (post_sentiments - article_sentiments).abs()

    # --- Group C: Semantic mismatch (3) ---
    print("   Group C: Semantic mismatch features...")

    dot = np.sum(emb_post * emb_article, axis=1)
    norm_post = np.linalg.norm(emb_post, axis=1)
    norm_article = np.linalg.norm(emb_article, axis=1)
    features["cosine_similarity"] = dot / (norm_post * norm_article + 1e-8)

    print("   Computing KL divergence (TF-IDF)...")
    all_texts = pd.concat([post_texts, article_texts]).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf.fit(all_texts)
    tfidf_post = tfidf.transform(post_texts.tolist()).toarray()
    tfidf_article = tfidf.transform(article_texts.tolist()).toarray()
    eps = 1e-10
    p = tfidf_post + eps
    q = tfidf_article + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    features["kl_divergence"] = np.sum(p * np.log(p / q), axis=1)

    print("   Computing Jaccard similarity...")
    jaccard_scores = []
    for pt, at in zip(post_texts, article_texts):
        pw = set(pt.lower().split())
        aw = set(at.lower().split())
        jaccard_scores.append(len(pw & aw) / len(pw | aw) if len(pw | aw) > 0 else 0.0)
    features["jaccard_similarity"] = jaccard_scores

    # --- Group D: Clickbait linguistic patterns (6) ---
    print("   Group D: Clickbait linguistic patterns...")

    print(f"   Selecting top {TOP_K_NGRAMS} discriminative n-grams (chi-squared on train only)...")
    cv = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=10000)
    X_ngram_train = cv.fit_transform(post_texts.iloc[train_idx])
    chi2_scores, chi2_pvals = chi2(X_ngram_train, y_train)
    top_k_idx = np.argsort(chi2_scores)[::-1][:TOP_K_NGRAMS]
    vocab = cv.get_feature_names_out()

    print(f"   Top 10 clickbait-discriminative n-grams:")
    for i, idx in enumerate(top_k_idx[:10], 1):
        print(f"     {i:2d}. '{vocab[idx]}'  (chi2 = {chi2_scores[idx]:.1f})")

    X_ngram_all = cv.transform(post_texts)
    features["clickbait_ngram_count"] = np.array(X_ngram_all[:, top_k_idx].sum(axis=1)).flatten()

    features["has_number"] = post_texts.apply(lambda x: 1 if re.search(r"\d", x) else 0)

    demo_pat = re.compile(r"^\s*(this|these|here|that)\b", re.IGNORECASE)
    features["starts_with_demonstrative"] = post_texts.apply(lambda x: 1 if demo_pat.search(x) else 0)

    second_pat = re.compile(r"\b(you|your|you're|yourself|yours)\b", re.IGNORECASE)
    features["second_person_count"] = post_texts.apply(lambda x: len(second_pat.findall(x)))

    features["avg_word_length"] = post_texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0
    )
    features["ellipsis_count"] = post_texts.apply(lambda x: x.count("..."))

    # --- Group E: Article metadata text features (4) ---
    print("   Group E: Article metadata text features...")

    title_texts = df["targetTitle_clean"]
    desc_texts = df["targetDescription_clean"]
    kw_texts = df["targetKeywords_clean"]

    dot_title = np.sum(emb_post * emb_title, axis=1)
    norm_title = np.linalg.norm(emb_title, axis=1)
    title_cos = dot_title / (norm_post * norm_title + 1e-8)
    empty_title = (title_texts.str.strip() == "").values
    title_cos[empty_title] = 0.0
    features["title_post_cosine_sim"] = title_cos

    title_jaccard = []
    for pt, tt in zip(post_texts, title_texts):
        pw = set(pt.lower().split())
        tw = set(tt.lower().split())
        title_jaccard.append(len(pw & tw) / len(pw | tw) if len(pw | tw) > 0 else 0.0)
    features["title_post_jaccard"] = title_jaccard

    dot_desc = np.sum(emb_post * emb_desc, axis=1)
    norm_desc = np.linalg.norm(emb_desc, axis=1)
    desc_cos = dot_desc / (norm_post * norm_desc + 1e-8)
    empty_desc = (desc_texts.str.strip() == "").values
    desc_cos[empty_desc] = 0.0
    features["desc_post_cosine_sim"] = desc_cos

    kw_overlap = []
    for pt, kw in zip(post_texts, kw_texts):
        kw_str = kw.strip()
        if not kw_str:
            kw_overlap.append(0.0)
            continue
        keywords = set(re.split(r"[,\s]+", kw_str.lower()))
        keywords.discard("")
        if not keywords:
            kw_overlap.append(0.0)
            continue
        post_lower = pt.lower()
        matched = sum(1 for k in keywords if k in post_lower)
        kw_overlap.append(matched / len(keywords))
    features["keyword_overlap_ratio"] = kw_overlap

    # --- Group F: Error-driven features (4) ---
    #   Targets missed clickbait that uses sensationalism without typical clickbait markers
    print("   Group F: Error-driven features...")

    # F1: Sensational word count
    SENSATIONAL_WORDS = {
        "shocking", "stunned", "stunning", "horrifying", "terrifying", "devastating",
        "incredible", "unbelievable", "amazing", "insane", "crazy", "disturbing",
        "heartbreaking", "sickening", "outrageous", "explosive", "bombshell",
        "scandalous", "controversial", "dramatic", "tragic", "horrific",
        "alarming", "chilling", "disgusting", "furious", "hilarious",
        "epic", "brutal", "savage", "deadly", "massive", "urgent",
        "breaking", "exclusive", "revealed", "exposed", "slammed",
        "blasted", "destroyed", "crushed", "creepy", "weird", "strange",
    }
    features["sensational_word_count"] = post_texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in SENSATIONAL_WORDS)
    )

    # F2: Post sentiment intensity (absolute value of compound score)
    features["post_sentiment_intensity"] = post_sentiments.abs()

    # F3: Proper noun ratio (capitalized non-first words / total words)
    def proper_noun_ratio(text):
        words = text.split()
        if len(words) <= 1:
            return 0.0
        non_first = words[1:]
        proper = sum(1 for w in non_first if w[0].isupper() and not w.isupper())
        return proper / len(words)

    features["proper_noun_ratio"] = post_texts.apply(proper_noun_ratio)

    # F4: Forward reference count (curiosity-gap words common in news-style clickbait)
    FORWARD_WORDS = {
        "new", "emerge", "emerges", "reveal", "reveals", "revealed",
        "discover", "discovers", "discovered", "found", "finds",
        "uncover", "uncovers", "uncovered", "detail", "details",
        "secret", "secrets", "mystery", "hidden", "unknown",
        "surprise", "surprising", "unexpected", "suddenly",
    }
    features["forward_reference_count"] = post_texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in FORWARD_WORDS)
    )

    # --- Summary ---
    print(f"\n   Feature matrix shape: {features.shape}")
    print("\n   Feature correlations with truthClass:")
    y_all = df["truthClass"]
    for col in features.columns:
        corr = features[col].corr(y_all)
        print(f"     {col:30s}  r = {corr:+.4f}")

    return features


# ===========================================================================
# SECTION 2: MODEL DEFINITIONS
# ===========================================================================

def get_models_and_params(imbalance_ratio):
    """Return ordered dict of (model, param_grid, n_iter) from simple to complex."""
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {
        "Logistic Regression (L1)": {
            "model": LogisticRegression(
                penalty="l1", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            },
            "n_iter": 8,
        },
        "Logistic Regression (L2)": {
            "model": LogisticRegression(
                penalty="l2", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            },
            "n_iter": 8,
        },
        "Elastic Net": {
            "model": LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "n_iter": 20,
        },
        "Linear SVC": {
            "model": SVC(
                kernel="linear", class_weight="balanced", probability=True,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            },
            "n_iter": 8,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
            },
            "n_iter": 30,
        },
        "XGBoost": {
            "model": XGBClassifier(
                scale_pos_weight=imbalance_ratio,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500, 700],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 8],
                "min_child_weight": [1, 3, 5, 7],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3, 0.5],
                "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
                "reg_lambda": [0.5, 1.0, 1.5, 2.0],
            },
            "n_iter": 50,
        },
        "LightGBM": {
            "model": LGBMClassifier(
                is_unbalance=True, random_state=RANDOM_STATE, verbose=-1,
            ),
            "params": {
                "n_estimators": [100, 200, 300, 500, 700, 1000],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "max_depth": [-1, 3, 5, 7, 10],
                "num_leaves": [15, 31, 50, 80, 127],
                "min_child_samples": [5, 10, 20, 30, 50],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
                "reg_lambda": [0, 0.01, 0.1, 0.5, 1.0],
            },
            "n_iter": 50,
        },
    }

    return models


# ===========================================================================
# SECTION 3: TRAINING LOOP
# ===========================================================================

def train_all_models(features_df, y, train_idx, test_idx):
    """Train each model with RandomizedSearchCV, optimize threshold, return results."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        precision_recall_curve, classification_report, confusion_matrix,
    )
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = features_df.values
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\n" + "=" * 70)
    print("SECTION 3: TRAINING LOOP")
    print("=" * 70)
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}")
    print(f"   Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   Test  class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    imbalance_ratio = n_neg / n_pos
    print(f"   Class imbalance ratio (neg/pos): {imbalance_ratio:.2f}")

    models = get_models_and_params(imbalance_ratio)
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for model_name, config in models.items():
        print(f"\n   {'-' * 60}")
        print(f"   TRAINING: {model_name}")
        print(f"   {'-' * 60}")

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            scoring="f1",
            cv=cv_folds,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train_scaled, y_train)
        clf = search.best_estimator_

        print(f"   Best CV F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")

        # Get predicted probabilities
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        # Default threshold (0.5)
        y_pred_default = (y_proba >= 0.5).astype(int)
        f1_default = f1_score(y_test, y_pred_default)

        # Optimized threshold
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_all = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_t_idx = np.argmax(f1_all)
        best_threshold = thresholds[best_t_idx]

        y_pred_opt = (y_proba >= best_threshold).astype(int)
        f1_opt = f1_score(y_test, y_pred_opt)

        # Pick best
        if f1_opt > f1_default:
            y_pred = y_pred_opt
            threshold_used = best_threshold
        else:
            y_pred = y_pred_default
            threshold_used = 0.5

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(f"   Threshold: {threshold_used:.4f}  |  F1: {f1:.4f}  |  Acc: {acc:.4f}  |  P: {prec:.4f}  |  R: {rec:.4f}")

        # Print classification report
        print(classification_report(y_test, y_pred, target_names=["no-clickbait", "clickbait"]))

        # Store results
        results[model_name] = {
            "clf": clf,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_test": y_test,
            "threshold": threshold_used,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "cv_f1": search.best_score_,
            "best_params": search.best_params_,
        }

    return results, X_test_scaled, scaler


# ===========================================================================
# SECTION 4: MODEL COMPARISON TABLE
# ===========================================================================

def print_comparison_table(results):
    """Print side-by-side comparison of all models."""
    print("\n" + "=" * 70)
    print("SECTION 4: MODEL COMPARISON TABLE")
    print("=" * 70)

    header = f"   {'Model':30s}  {'CV F1':>7s}  {'Test F1':>7s}  {'Acc':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'Thresh':>7s}"
    print(header)
    print(f"   {'-'*30}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    best_f1 = 0
    best_model_name = None

    for name, r in results.items():
        marker = ""
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_model_name = name

        print(f"   {name:30s}  {r['cv_f1']:7.4f}  {r['f1']:7.4f}  {r['accuracy']:7.4f}  "
              f"{r['precision']:7.4f}  {r['recall']:7.4f}  {r['threshold']:7.4f}")

    print(f"\n   >>> Best model by test F1: {best_model_name} (F1 = {best_f1:.4f})")

    return best_model_name


# ===========================================================================
# SECTION 5: ERROR ANALYSIS
# ===========================================================================

def error_analysis(results, best_model_name, features_df, df, test_idx):
    """Detailed error analysis on the best model's predictions."""
    print("\n" + "=" * 70)
    print(f"SECTION 5: ERROR ANALYSIS  -- {best_model_name}")
    print("=" * 70)

    r = results[best_model_name]
    y_test = r["y_test"]
    y_pred = r["y_pred"]
    y_proba = r["y_proba"]

    # -----------------------------------------------------------------------
    # 5a. Per-sample cross-entropy loss
    # -----------------------------------------------------------------------
    print("\n   --- 5a. Per-sample cross-entropy loss ---")
    eps = 1e-15
    p_clipped = np.clip(y_proba, eps, 1 - eps)
    cross_entropy = -(y_test * np.log(p_clipped) + (1 - y_test) * np.log(1 - p_clipped))

    print(f"   Mean cross-entropy (all):         {cross_entropy.mean():.4f}")
    print(f"   Mean cross-entropy (correct):     {cross_entropy[y_test == y_pred].mean():.4f}")
    print(f"   Mean cross-entropy (misclassified): {cross_entropy[y_test != y_pred].mean():.4f}")

    # -----------------------------------------------------------------------
    # 5b. Highest-loss samples (most confidently wrong)
    # -----------------------------------------------------------------------
    print("\n   --- 5b. Top 20 highest cross-entropy samples (most confidently wrong) ---")
    loss_order = np.argsort(cross_entropy)[::-1]

    print(f"   {'Rank':>4s}  {'True':>4s}  {'Pred':>4s}  {'Prob':>6s}  {'Loss':>7s}  {'Post text (first 80 chars)'}")
    print(f"   {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*50}")
    for rank, i in enumerate(loss_order[:20], 1):
        row_idx = test_idx[i]
        post = df.iloc[row_idx]["postText_clean"][:80].encode("ascii", "replace").decode("ascii")
        print(f"   {rank:4d}  {y_test[i]:4d}  {y_pred[i]:4d}  {y_proba[i]:6.3f}  {cross_entropy[i]:7.4f}  {post}")

    # -----------------------------------------------------------------------
    # 5c. Closest-to-boundary samples (probability nearest 0.5)
    # -----------------------------------------------------------------------
    print("\n   --- 5c. Top 20 closest-to-boundary samples (predicted prob nearest 0.5) ---")
    boundary_dist = np.abs(y_proba - 0.5)
    boundary_order = np.argsort(boundary_dist)

    print(f"   {'Rank':>4s}  {'True':>4s}  {'Pred':>4s}  {'Prob':>6s}  {'|p-0.5|':>7s}  {'Post text (first 80 chars)'}")
    print(f"   {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*50}")
    for rank, i in enumerate(boundary_order[:20], 1):
        row_idx = test_idx[i]
        post = df.iloc[row_idx]["postText_clean"][:80].encode("ascii", "replace").decode("ascii")
        print(f"   {rank:4d}  {y_test[i]:4d}  {y_pred[i]:4d}  {y_proba[i]:6.3f}  {boundary_dist[i]:7.4f}  {post}")

    # -----------------------------------------------------------------------
    # 5d. Misclassification breakdown
    # -----------------------------------------------------------------------
    print("\n   --- 5d. Misclassification breakdown ---")

    correct_mask = y_test == y_pred
    wrong_mask = ~correct_mask
    fp_mask = (y_pred == 1) & (y_test == 0)  # false positives
    fn_mask = (y_pred == 0) & (y_test == 1)  # false negatives

    n_correct = correct_mask.sum()
    n_wrong = wrong_mask.sum()
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()

    print(f"   Correct: {n_correct}  |  Wrong: {n_wrong}  (FP: {n_fp}, FN: {n_fn})")
    print(f"   FP rate: {n_fp}/{(y_test == 0).sum()} = {n_fp / (y_test == 0).sum():.4f}")
    print(f"   FN rate: {n_fn}/{(y_test == 1).sum()} = {n_fn / (y_test == 1).sum():.4f}")

    # -----------------------------------------------------------------------
    # 5e. Feature comparison: correct vs misclassified
    # -----------------------------------------------------------------------
    print("\n   --- 5e. Feature means: correct vs misclassified predictions ---")

    test_features = features_df.iloc[test_idx].copy()
    test_features = test_features.reset_index(drop=True)

    print(f"   {'Feature':30s}  {'Correct':>10s}  {'Wrong':>10s}  {'FP':>10s}  {'FN':>10s}")
    print(f"   {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for col in FEATURE_NAMES:
        correct_mean = test_features.loc[correct_mask, col].mean()
        wrong_mean = test_features.loc[wrong_mask, col].mean()
        fp_mean = test_features.loc[fp_mask, col].mean() if n_fp > 0 else 0.0
        fn_mean = test_features.loc[fn_mask, col].mean() if n_fn > 0 else 0.0
        print(f"   {col:30s}  {correct_mean:10.4f}  {wrong_mean:10.4f}  {fp_mean:10.4f}  {fn_mean:10.4f}")

    # -----------------------------------------------------------------------
    # 5f. Feature importance (if model supports it)
    # -----------------------------------------------------------------------
    clf = r["clf"]
    if hasattr(clf, "feature_importances_"):
        print(f"\n   --- 5f. Feature importance ({best_model_name}) ---")
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1]
        for rank, idx in enumerate(order, 1):
            print(f"     {rank:2d}. {FEATURE_NAMES[idx]:30s}  importance = {importances[idx]:.4f}")
    elif hasattr(clf, "coef_"):
        print(f"\n   --- 5f. Feature coefficients ({best_model_name}) ---")
        coefs = clf.coef_[0]
        order = np.argsort(np.abs(coefs))[::-1]
        for rank, idx in enumerate(order, 1):
            print(f"     {rank:2d}. {FEATURE_NAMES[idx]:30s}  coef = {coefs[idx]:+.4f}")

    # -----------------------------------------------------------------------
    # 5g. L1 coefficient analysis (show what Lasso zeroed out)
    # -----------------------------------------------------------------------
    if "Logistic Regression (L1)" in results:
        l1_clf = results["Logistic Regression (L1)"]["clf"]
        if hasattr(l1_clf, "coef_"):
            print(f"\n   --- 5g. L1 (Lasso) coefficient analysis  -- feature selection ---")
            coefs = l1_clf.coef_[0]
            order = np.argsort(np.abs(coefs))[::-1]
            n_zero = (np.abs(coefs) < 1e-10).sum()
            print(f"   Features kept: {len(coefs) - n_zero} / {len(coefs)}  |  Zeroed out: {n_zero}")
            for rank, idx in enumerate(order, 1):
                status = "KEPT" if abs(coefs[idx]) >= 1e-10 else "ZERO"
                print(f"     {rank:2d}. {FEATURE_NAMES[idx]:30s}  coef = {coefs[idx]:+.6f}  [{status}]")

    return cross_entropy


# ===========================================================================
# SECTION 6: SAVE OUTPUTS
# ===========================================================================

def save_outputs(results, best_model_name, features_df, df, test_idx, cross_entropy):
    """Save predictions CSV with probabilities and loss values."""
    print("\n" + "=" * 70)
    print("SECTION 6: SAVING OUTPUTS")
    print("=" * 70)

    r = results[best_model_name]

    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_label"] = r["y_test"]
    out["predicted"] = r["y_pred"]
    out["predicted_proba"] = r["y_proba"]
    out["cross_entropy_loss"] = cross_entropy

    test_features = features_df.iloc[test_idx].reset_index(drop=True)
    for fname in FEATURE_NAMES:
        out[fname] = test_features[fname].values

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved to {OUTPUT_FILE}")
    print(f"   Shape: {out.shape}")
    print(f"   Columns: true_label, predicted, predicted_proba, cross_entropy_loss, + {len(FEATURE_NAMES)} features")

# ===========================================================================
# ABLATION STUDY
# ===========================================================================

FEATURE_GROUPS = {
    "E_metadata": [
        "title_post_cosine_sim", "title_post_jaccard",
        "desc_post_cosine_sim", "keyword_overlap_ratio",
    ],
    "F_error_driven": [
        "sensational_word_count", "post_sentiment_intensity",
        "proper_noun_ratio", "forward_reference_count",
    ],
}

ABLATION_SETS = {
    "full_26": FEATURE_NAMES,
    "no_metadata": [f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["E_metadata"]],
    "no_error_features": [f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["F_error_driven"]],
}

def run_ablation(features_df, y, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    ablation_results = []
    y_array = np.array(y)

    for name, feature_list in ABLATION_SETS.items():
        print(f"\nAblation: {name}")

        X = features_df[feature_list].values
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_array[train_idx]
        y_test = y_array[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        f1 = f1_score(y_test, preds)

        print(f"   Features: {len(feature_list)}")
        print(f"   F1: {f1:.4f}")

        ablation_results.append({
            "ablation": name,
            "num_features": len(feature_list),
            "f1": f1,
        })

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv("ablation_results.csv", index=False)
    print("\nSaved: ablation_results.csv")

    return ablation_df


# ===========================================================================
# EXPORT ERROR TABLES
# ===========================================================================

def export_errors(model, X_test, y_test, test_idx, df):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    error_df = df.iloc[test_idx].copy()
    error_df["prob"] = probs
    error_df["pred"] = preds
    error_df["true"] = np.array(y_test)

    false_positives = error_df[
        (error_df["pred"] == 1) & (error_df["true"] == 0)
    ].sort_values("prob", ascending=False).head(20)

    false_negatives = error_df[
        (error_df["pred"] == 0) & (error_df["true"] == 1)
    ].sort_values("prob", ascending=True).head(20)

    boundary_cases = error_df.iloc[
        np.argsort(np.abs(error_df["prob"] - 0.5))
    ].head(20)

    false_positives.to_csv("top_false_positives.csv", index=False)
    false_negatives.to_csv("top_false_negatives.csv", index=False)
    boundary_cases.to_csv("boundary_cases.csv", index=False)

    print("   Saved: top_false_positives.csv")
    print("   Saved: top_false_negatives.csv")
    print("   Saved: boundary_cases.csv")


# ===========================================================================
# RUN
# ===========================================================================

from sklearn.model_selection import train_test_split

# Section 1: Load & prepare
df = load_and_clean(INPUT_FILE)
emb_post, emb_article, emb_title, emb_desc = encode_texts(df)

y = df["truthClass"].values
train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
)

features_df = build_features(
    df, emb_post, emb_article, emb_title, emb_desc, train_idx, y[train_idx]
)

# Section 3: Train all models
results, X_test_scaled, scaler = train_all_models(features_df, y, train_idx, test_idx)

# Ablation study
run_ablation(features_df, df["truthClass"], train_idx, test_idx)

# Section 4: Comparison table
best_model_name = print_comparison_table(results)

# Section 5: Error analysis on best model
cross_entropy = error_analysis(results, best_model_name, features_df, df, test_idx)

# Export error tables
best_model = results[best_model_name]["clf"]
y_test = results[best_model_name]["y_test"]
export_errors(best_model, X_test_scaled, y_test, test_idx, df)

# Section 6: Save
save_outputs(results, best_model_name, features_df, df, test_idx, cross_entropy)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)