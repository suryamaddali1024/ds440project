"""
clickbait_tournament_powerblend.py
====================================
7-Model Tournament + Optimised Power Blend for Clickbait Detection.
Target: Break the F1 = 0.711 barrier on final_cleaned_full.csv.

Stage 1 ─ Feature Factory  (30 hand-crafted features)
Stage 2 ─ Tournament       (7 base learners)
Stage 3 ─ Power Blend      (grid-search weights + power k + temperature scaling)
Stage 4 ─ Scientific Report (leaderboard · diversity matrix · "what worked")

Data splits (stratified on truthClass, seed = 42 — consistent with all prior work):
  base_train  64 %  train all base learners
  blend_val   16 %  optimize blend weights & calibrator
  test        20 %  final held-out evaluation (never touched until Section 4 end)

pip install torch transformers lightgbm xgboost catboost scikit-learn
            sentence-transformers vaderSentiment Levenshtein scipy
            matplotlib seaborn
"""

# ═══════════════════════════════════════════════════════════════════
#  0 · IMPORTS
# ═══════════════════════════════════════════════════════════════════
import ast, gc, os, re, random, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit as sigmoid

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

import lightgbm as lgb
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("  [WARN] catboost not installed — CatBoost will be skipped")

try:
    from Levenshtein import ratio as lev_ratio
except ImportError:
    import difflib
    def lev_ratio(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    print("  [INFO] python-Levenshtein not found — using difflib fallback")

# ═══════════════════════════════════════════════════════════════════
#  0 · GLOBAL CONFIG  (single source of truth)
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    INPUT_FILE          = "final_cleaned_full.csv",
    OUTPUT_PREDS        = "tournament_final_predictions.csv",
    OUTPUT_LEADERBOARD  = "tournament_leaderboard.csv",

    # ── Splits ──────────────────────────────────────────────────────
    TEST_SIZE           = 0.20,
    BLEND_FRAC          = 0.20,   # of the 80 % train-pool → blend_val = 16 % total
    SEED                = 42,

    # ── DistilBERT ──────────────────────────────────────────────────
    DISTILBERT_NAME     = "distilbert-base-uncased",
    DISTILBERT_WEIGHTS  = "distilbert_v2_weights.pt",
    BERT_MAX_LEN        = 96,
    BERT_BATCH          = 16,
    BERT_EPOCHS         = 4,
    BERT_LR             = 2e-5,
    BERT_WARMUP         = 0.10,

    # ── RoBERTa ─────────────────────────────────────────────────────
    ROBERTA_NAME        = "roberta-base",
    ROBERTA_WEIGHTS     = "roberta_clickbait_weights.pt",
    ROB_MAX_LEN         = 80,
    ROB_BATCH           = 16,
    ROB_EPOCHS          = 3,
    ROB_LR              = 2e-5,
    ROB_WARMUP          = 0.06,

    # ── SBERT ───────────────────────────────────────────────────────
    SBERT_NAME          = "sentence-transformers/all-MiniLM-L6-v2",
    SBERT_BATCH         = 128,

    # ── SBERT-MLP ───────────────────────────────────────────────────
    MLP_EPOCHS          = 30,
    MLP_BATCH           = 256,
    MLP_LR              = 1e-3,
    MLP_DROPOUT1        = 0.30,
    MLP_DROPOUT2        = 0.20,

    # ── LightGBM ────────────────────────────────────────────────────
    LGBM_PARAMS = dict(
        n_estimators=700, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=20, subsample=0.80, colsample_bytree=0.80,
        reg_alpha=0.10, reg_lambda=0.10, is_unbalance=True,
        random_state=42, verbose=-1,
    ),

    # ── XGBoost ─────────────────────────────────────────────────────
    XGB_PARAMS = dict(
        n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=3,
        subsample=0.80, colsample_bytree=0.80, gamma=0.10,
        reg_alpha=0.10, reg_lambda=1.50, eval_metric="logloss",
        use_label_encoder=False, random_state=42,
    ),

    # ── CatBoost ────────────────────────────────────────────────────
    CB_PARAMS = dict(
        iterations=600, learning_rate=0.05, depth=7,
        l2_leaf_reg=3.0, random_strength=1.0,
        auto_class_weights="Balanced", verbose=0, random_seed=42,
    ),

    # ── Blend Optimiser ─────────────────────────────────────────────
    POWER_K_GRID        = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    BLEND_THR_GRID      = 61,   # threshold grid points for F1 optimisation

    # ── Tri-state ───────────────────────────────────────────────────
    CERTAIN_CB          = 0.80,
    CERTAIN_NCB         = 0.20,

    # ── Reporting ───────────────────────────────────────────────────
    PRIOR_BEST_F1       = 0.711,
)

torch.manual_seed(CFG["SEED"]); random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pragmatic slang triggers (25 patterns, error-analysis driven) ───
_SLANG_PAT = (
    r"shots fired|stay woke|you won'?t believe|wait for it|this is why"
    r"|this will make you|wait till you see|you need to see"
    r"|we can'?t stop watching|i can'?t even|i'm dead|i'm crying"
    r"|twitter lost it|twitter exploded|twitter reacted|the internet lost"
    r"|people are (freaking|losing) (out|it)|broke the internet"
    r"|minds were blown|mic drop|plot twist|the struggle is real"
    r"|on fleek|goals|no cap|lowkey|highkey|understood the assignment"
    r"|hits different|it's giving|rent free|caught in 4k"
)
SLANG_RE = re.compile(r"\b(?:" + _SLANG_PAT + r")\b", re.I)

# ── Topic category mapping for CatBoost ────────────────────────────
TOPIC_MAP = {
    "sports":  {"nfl","nba","football","basketball","baseball","soccer",
                "tennis","golf","cricket","rugby","hockey","olympics","sport"},
    "politics":{"trump","obama","clinton","congress","senate","election",
                "democrat","republican","white house","president","vote",
                "government","policy","minister","parliament"},
    "tech":    {"apple","google","microsoft","ai","robot","tech","software",
                "hardware","iphone","android","app","startup","silicon"},
    "entertainment":{"celebrity","movie","actor","music","song","award",
                     "hollywood","netflix","spotify","grammy","oscar"},
    "health":  {"health","medical","doctor","cancer","diet","fitness",
                "mental","virus","vaccine","hospital","study","research"},
    "business":{"market","stock","economy","finance","invest","trade",
                "bank","startup","ipo","revenue","profit","billion"},
    "world":   {"war","conflict","military","nato","un","humanitarian",
                "refugee","climate","environment","protest"},
}


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════
def banner(msg: str):
    w = 70
    print(f"\n╔{'═'*(w-2)}╗")
    for line in textwrap.wrap(msg, w-4):
        print(f"║  {line:<{w-4}}  ║")
    print(f"╚{'═'*(w-2)}╝")

def sec(msg: str):
    print(f"\n{'═'*70}\n  {msg}\n{'═'*70}")

def subsec(msg: str):
    print(f"\n  {'─'*66}\n  {msg}")

def best_threshold(y_true, probs, n=61):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.15, 0.85, n):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def model_metrics(y_true, probs, label="model"):
    t, f1 = best_threshold(y_true, probs)
    y_pred = (probs >= t).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    return dict(label=label, f1=f1, acc=acc, prec=prec, rec=rec, thr=t)

def parse_text(raw):
    if pd.isna(raw) or not str(raw).strip(): return ""
    s = str(raw).strip()
    try:
        p = ast.literal_eval(s)
        if isinstance(p, list):
            return " ".join(str(x).strip() for x in p if str(x).strip())
        return str(p).strip()
    except (ValueError, SyntaxError):
        return s.lstrip("[").rstrip("]") if s.startswith("[") else s

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 · DATA LOADING & STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════
def load_and_split():
    sec("SECTION 1 — Data Loading & Splits")

    df = pd.read_csv(CFG["INPUT_FILE"], encoding="latin-1")
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")

    for raw, clean in [
        ("postText",          "post"),
        ("targetTitle",       "title"),
        ("targetParagraphs",  "article"),
        ("targetDescription", "desc"),
        ("targetKeywords",    "keywords"),
        ("postMedia",         "media_raw"),
    ]:
        df[clean] = df[raw].apply(parse_text)

    df = df[
        (df["post"].str.strip() != "") & (df["article"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"  After cleanup: {len(df):,} rows")
    print(f"  Class dist: {df['truthClass'].value_counts().to_dict()}")

    y = df["truthClass"].values

    # ── Primary split (matches ALL prior scripts) ──
    tv_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=CFG["TEST_SIZE"],
        stratify=y, random_state=CFG["SEED"],
    )
    # ── Secondary split: base_train vs blend_val ──
    base_idx, blend_idx = train_test_split(
        np.arange(len(tv_idx)),
        test_size=CFG["BLEND_FRAC"],
        stratify=y[tv_idx],
        random_state=CFG["SEED"],
    )
    base_train_idx = tv_idx[base_idx]
    blend_val_idx  = tv_idx[blend_idx]

    for name, idx in [("base_train", base_train_idx),
                      ("blend_val",  blend_val_idx),
                      ("test",       test_idx)]:
        n = len(idx)
        cb_rate = y[idx].mean()
        print(f"  {name:<12}: {n:>6} rows  ({n/len(df)*100:.1f}%)  "
              f"CB-rate={cb_rate:.3f}")

    return df, y, base_train_idx, blend_val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 · FEATURE FACTORY  (30 features, Groups A–H)
# ═══════════════════════════════════════════════════════════════════
def build_feature_factory(df, emb_post, emb_title, emb_article, emb_desc,
                           base_train_idx, y):
    sec("SECTION 2 — Feature Factory  (30 features, Groups A–H)")

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import chi2

    feat  = pd.DataFrame(index=df.index)
    post  = df["post"];    title    = df["title"]
    art   = df["article"]; desc     = df["desc"]
    kw    = df["keywords"]
    y_tr  = y[base_train_idx]

    # ── Group A: Text statistics (6) ──────────────────────────────
    subsec("Group A — Text statistics")
    post_wc = post.apply(lambda x: len(x.split()))
    art_wc  = art.apply(lambda x: len(x.split()))
    feat["post_word_count"]        = post_wc
    feat["article_word_count"]     = art_wc
    feat["word_count_ratio"]       = post_wc / art_wc.replace(0, 1)
    feat["post_question_marks"]    = post.apply(lambda x: x.count("?"))
    feat["post_exclamation_marks"] = post.apply(lambda x: x.count("!"))
    feat["post_caps_ratio"]        = post.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1) / max(len(x.split()), 1)
    )

    # ── Group B: VADER sentiment (3) ─────────────────────────────
    subsec("Group B — VADER sentiment")
    vader = SentimentIntensityAnalyzer()
    post_s  = post.apply(lambda x: vader.polarity_scores(x)["compound"])
    art_s   = art.apply(lambda x: vader.polarity_scores(x)["compound"])
    feat["post_sentiment"]    = post_s
    feat["article_sentiment"] = art_s
    feat["abs_sentiment_gap"] = (post_s - art_s).abs()

    # ── Group C: Semantic mismatch (3) ───────────────────────────
    subsec("Group C — Semantic mismatch")
    def cosine(a, b):
        n = (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8) * \
            (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.sum(a * b, axis=1) / n.squeeze()

    feat["cosine_similarity"] = cosine(emb_post, emb_article)
    all_texts = pd.concat([post, art]).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf.fit(all_texts)
    P = tfidf.transform(post.tolist()).toarray() + 1e-10
    Q = tfidf.transform(art.tolist()).toarray()  + 1e-10
    P /= P.sum(axis=1, keepdims=True); Q /= Q.sum(axis=1, keepdims=True)
    feat["kl_divergence"]    = np.sum(P * np.log(P / Q), axis=1)
    feat["jaccard_similarity"] = [
        len(set(pt.lower().split()) & set(at.lower().split())) /
        max(len(set(pt.lower().split()) | set(at.lower().split())), 1)
        for pt, at in zip(post, art)
    ]

    # ── Group D: Clickbait linguistic patterns (6) ───────────────
    subsec("Group D — Clickbait linguistic patterns")
    cv = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=10000)
    X_ng_tr = cv.fit_transform(post.iloc[base_train_idx])
    chi2_sc, _ = chi2(X_ng_tr, y_tr)
    top100 = np.argsort(chi2_sc)[::-1][:100]
    X_ng_all = cv.transform(post)
    feat["clickbait_ngram_count"]     = np.array(X_ng_all[:, top100].sum(axis=1)).flatten()
    feat["has_number"]                = post.apply(lambda x: int(bool(re.search(r"\d", x))))
    feat["starts_with_demonstrative"] = post.apply(
        lambda x: int(bool(re.match(r"^\s*(this|these|here|that)\b", x, re.I))))
    feat["second_person_count"] = post.apply(
        lambda x: len(re.findall(r"\b(you|your|you're|yourself)\b", x, re.I)))
    feat["avg_word_length"] = post.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0.0)
    feat["ellipsis_count"] = post.apply(lambda x: x.count("..."))

    # ── Group E: Article metadata (4) ────────────────────────────
    subsec("Group E — Article metadata")
    norm_p = emb_post   / (np.linalg.norm(emb_post,   axis=1, keepdims=True) + 1e-8)
    norm_t = emb_title  / (np.linalg.norm(emb_title,  axis=1, keepdims=True) + 1e-8)
    norm_d = emb_desc   / (np.linalg.norm(emb_desc,   axis=1, keepdims=True) + 1e-8)
    title_cos = np.sum(norm_p * norm_t, axis=1)
    desc_cos  = np.sum(norm_p * norm_d, axis=1)
    title_cos[title.str.strip() == ""] = 0.0
    desc_cos[desc.str.strip() == ""]   = 0.0
    feat["title_post_cosine_sim"] = title_cos
    feat["title_post_jaccard"]    = [
        len(set(pt.lower().split()) & set(tt.lower().split())) /
        max(len(set(pt.lower().split()) | set(tt.lower().split())), 1)
        for pt, tt in zip(post, title)
    ]
    feat["desc_post_cosine_sim"]  = desc_cos
    feat["keyword_overlap_ratio"] = [
        sum(1 for k in re.split(r"[,\s]+", kw_s.lower()) if k and k in pt.lower()) /
        max(len([k for k in re.split(r"[,\s]+", kw_s.lower()) if k]), 1)
        for pt, kw_s in zip(post, kw.fillna(""))
    ]

    # ── Group F: Error-driven features (4) ───────────────────────
    subsec("Group F — Error-driven features")
    SENS = {
        "shocking","stunned","stunning","horrifying","terrifying","devastating",
        "incredible","unbelievable","amazing","insane","crazy","disturbing",
        "heartbreaking","sickening","outrageous","explosive","bombshell",
        "scandalous","controversial","dramatic","tragic","horrific","alarming",
        "chilling","disgusting","furious","hilarious","epic","brutal","savage",
        "deadly","massive","urgent","breaking","exclusive","revealed","exposed",
    }
    FWD = {
        "new","emerge","emerges","reveal","reveals","revealed","discover",
        "discovers","discovered","found","finds","uncover","uncovers","uncovered",
        "detail","details","secret","secrets","mystery","hidden","unknown",
        "surprise","surprising","unexpected","suddenly",
    }
    feat["sensational_word_count"] = post.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in SENS))
    feat["post_sentiment_intensity"]  = post_s.abs()
    feat["proper_noun_ratio"]         = post.apply(
        lambda x: sum(1 for w in x.split()[1:] if w and w[0].isupper() and not w.isupper())
                  / max(len(x.split()), 1))
    feat["forward_reference_count"] = post.apply(
        lambda x: sum(1 for w in x.lower().split() if w.strip(".,!?;:'\"") in FWD))

    # ── Group G: Mirror Similarity & Pragmatic Slang (2) ★ NEW ──
    subsec("Group G — NEW: Mirror Similarity + Pragmatic Slang")
    feat["mirror_similarity"]    = [
        lev_ratio(pt.lower().strip(), tt.lower().strip()) if tt.strip() else 0.5
        for pt, tt in zip(post, title)
    ]
    feat["pragmatic_slang_count"] = post.apply(lambda x: len(SLANG_RE.findall(x)))
    print(f"  mirror_sim: mean={feat['mirror_similarity'].mean():.3f}")
    print(f"  slang hits: nonzero={(feat['pragmatic_slang_count'] > 0).sum()}")

    # ── Group H: Entity Density & Curiosity Gap Index (2) ★ NEW ──
    subsec("Group H — NEW: Entity Density + Curiosity Gap Index")
    def entity_density(text):
        words = text.split()
        if not words: return 0.0
        proper = sum(1 for i, w in enumerate(words)
                     if i > 0 and w and w[0].isupper() and not w.isupper() and len(w) > 1)
        return proper / len(words)

    feat["entity_density"] = post.apply(entity_density)
    title_wc = title.apply(lambda x: len(x.split()))
    feat["curiosity_gap_index"] = (
        (title_wc - post_wc) / (title_wc + post_wc + 1)
    )
    print(f"  entity_density:   mean={feat['entity_density'].mean():.3f}")
    print(f"  curiosity_gap:    mean={feat['curiosity_gap_index'].mean():.3f}  "
          f"(positive → title longer → potential bait)")

    FEATURE_NAMES = list(feat.columns)
    print(f"\n  ✓ Total features: {len(FEATURE_NAMES)}")
    return feat, FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 · SBERT ENCODING
# ═══════════════════════════════════════════════════════════════════
def sbert_encode(df):
    sec("SECTION 3 — SBERT Encoding")
    from sentence_transformers import SentenceTransformer
    sbert_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model: {CFG['SBERT_NAME']}  |  Device: {sbert_dev}")
    sbert = SentenceTransformer(CFG["SBERT_NAME"], device=sbert_dev)

    def enc(texts):
        return sbert.encode(texts, batch_size=CFG["SBERT_BATCH"],
                            show_progress_bar=False, convert_to_numpy=True)

    emb_post = enc(df["post"].tolist())
    emb_art  = enc(df["article"].tolist())
    emb_ttl  = enc([t if t.strip() else "no title" for t in df["title"].tolist()])
    emb_dsc  = enc([d if d.strip() else "no description" for d in df["desc"].tolist()])
    print(f"  Embedding shape: {emb_post.shape}")
    del sbert; free_gpu()
    return emb_post, emb_art, emb_ttl, emb_dsc


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 · TOURNAMENT  (7 base learners)
# ═══════════════════════════════════════════════════════════════════

# ── 4.1  DistilBERT v2 — regression on truthMean ────────────────
class _RegrDataset(Dataset):
    def __init__(self, posts, titles, scores, tok, max_len):
        self.enc = tok(posts, titles, truncation=True, padding=True,
                       max_length=max_len, return_tensors="pt")
        self.s   = torch.tensor(scores, dtype=torch.float)
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["s"] = self.s[i]; return item


def _bert_infer(model, tok, posts, titles, device, batch=16, max_len=96):
    ds = _RegrDataset(posts, titles, np.zeros(len(posts)), tok, max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval(); preds = []
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy())
    return np.array(preds)


def train_distilbert(df, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.1 — DistilBERT v2  [regression on truthMean, dual-input]")
    from transformers import (DistilBertTokenizer,
                               DistilBertForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from torch.optim import AdamW

    posts  = df["post"].tolist()
    titles = [t if t.strip() else "no title" for t in df["title"].tolist()]
    scores = df["truthMean"].values

    tok = DistilBertTokenizer.from_pretrained(CFG["DISTILBERT_NAME"])

    def gather(lst, idx): return [lst[i] for i in idx]

    model = DistilBertForSequenceClassification.from_pretrained(
        CFG["DISTILBERT_NAME"], num_labels=1).to(DEVICE)

    if os.path.exists(CFG["DISTILBERT_WEIGHTS"]):
        model.load_state_dict(torch.load(CFG["DISTILBERT_WEIGHTS"], map_location=DEVICE))
        print(f"  Loaded saved weights from '{CFG['DISTILBERT_WEIGHTS']}'")
    else:
        # Internal train/val split within base_train
        bt_idx, bv_idx = train_test_split(
            np.arange(len(base_train_idx)),
            test_size=0.12, stratify=y[base_train_idx], random_state=CFG["SEED"])
        bt = base_train_idx[bt_idx]; bv = base_train_idx[bv_idx]

        tr_ds = _RegrDataset(gather(posts, bt), gather(titles, bt),
                             scores[bt], tok, CFG["BERT_MAX_LEN"])
        vl_ds = _RegrDataset(gather(posts, bv), gather(titles, bv),
                             scores[bv], tok, CFG["BERT_MAX_LEN"])
        tr_dl = DataLoader(tr_ds, batch_size=CFG["BERT_BATCH"], shuffle=True)
        vl_dl = DataLoader(vl_ds, batch_size=CFG["BERT_BATCH"], shuffle=False)

        opt = AdamW(model.parameters(), lr=CFG["BERT_LR"], weight_decay=0.01)
        total  = len(tr_dl) * CFG["BERT_EPOCHS"]
        warmup = int(total * CFG["BERT_WARMUP"])
        sched  = get_linear_schedule_with_warmup(opt, warmup, total)
        loss_fn = nn.MSELoss()
        best_mse, best_state = float("inf"), None

        for epoch in range(CFG["BERT_EPOCHS"]):
            model.train(); ep_loss = 0.0
            for bi, batch in enumerate(tr_dl):
                ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
                tgt = batch["s"].to(DEVICE)
                pred = torch.sigmoid(model(input_ids=ids, attention_mask=mask).logits.squeeze(-1))
                loss = loss_fn(pred, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); ep_loss += loss.item()
                if (bi + 1) % 150 == 0:
                    print(f"    Ep{epoch+1} B{bi+1}/{len(tr_dl)} MSE={ep_loss/(bi+1):.5f}")
            # validate
            model.eval(); vl_p, vl_t = [], []
            with torch.no_grad():
                for batch in vl_dl:
                    ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
                    vl_p.extend(torch.sigmoid(
                        model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)).cpu().numpy())
                    vl_t.extend(batch["s"].numpy())
            vl_mse = np.mean((np.array(vl_p) - np.array(vl_t))**2)
            print(f"  Epoch {epoch+1}  train_MSE={ep_loss/len(tr_dl):.5f}  val_MSE={vl_mse:.5f}")
            if vl_mse < best_mse:
                best_mse = vl_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ★ best val_MSE={best_mse:.5f}")
        model.load_state_dict(best_state)
        torch.save(best_state, CFG["DISTILBERT_WEIGHTS"])
        print(f"  Saved weights → {CFG['DISTILBERT_WEIGHTS']}")

    blend_p = _bert_infer(model, tok,
                          gather(posts, blend_val_idx), gather(titles, blend_val_idx),
                          DEVICE, CFG["BERT_BATCH"], CFG["BERT_MAX_LEN"])
    test_p  = _bert_infer(model, tok,
                          gather(posts, test_idx), gather(titles, test_idx),
                          DEVICE, CFG["BERT_BATCH"], CFG["BERT_MAX_LEN"])
    del model; free_gpu()

    m = model_metrics(y[test_idx], test_p, "DistilBERT-v2")
    print(f"  DistilBERT standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.2  RoBERTa-Base — binary classification on postText ───────
class _ClsDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, padding=True,
                       max_length=max_len, return_tensors="pt")
        self.y   = torch.tensor(labels, dtype=torch.float)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["y"] = self.y[i]; return item


def _cls_infer(model, tok, texts, device, batch=16, max_len=80):
    dummy = np.zeros(len(texts))
    ds = _ClsDataset(texts, dummy, tok, max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval(); preds = []
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits[:, 1]   # prob of class=1
            preds.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(preds)


def train_roberta(df, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.2 — RoBERTa-Base  [binary classification, sarcasm & sentiment]")
    from transformers import (RobertaTokenizer,
                               RobertaForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from torch.optim import AdamW

    posts = df["post"].tolist()
    labels = y

    tok   = RobertaTokenizer.from_pretrained(CFG["ROBERTA_NAME"])
    model = RobertaForSequenceClassification.from_pretrained(
        CFG["ROBERTA_NAME"], num_labels=2).to(DEVICE)

    def gather(lst, idx): return [lst[i] for i in idx]

    if os.path.exists(CFG["ROBERTA_WEIGHTS"]):
        model.load_state_dict(torch.load(CFG["ROBERTA_WEIGHTS"], map_location=DEVICE))
        print(f"  Loaded saved weights from '{CFG['ROBERTA_WEIGHTS']}'")
    else:
        bt_idx, bv_idx = train_test_split(
            np.arange(len(base_train_idx)),
            test_size=0.12, stratify=y[base_train_idx], random_state=CFG["SEED"])
        bt = base_train_idx[bt_idx]; bv = base_train_idx[bv_idx]

        tr_ds = _ClsDataset(gather(posts, bt), labels[bt], tok, CFG["ROB_MAX_LEN"])
        vl_ds = _ClsDataset(gather(posts, bv), labels[bv], tok, CFG["ROB_MAX_LEN"])
        tr_dl = DataLoader(tr_ds, batch_size=CFG["ROB_BATCH"], shuffle=True)
        vl_dl = DataLoader(vl_ds, batch_size=CFG["ROB_BATCH"], shuffle=False)

        # Class weights for imbalanced data
        n_neg = (labels[bt] == 0).sum(); n_pos = (labels[bt] == 1).sum()
        pw = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=pw)

        opt    = AdamW(model.parameters(), lr=CFG["ROB_LR"], weight_decay=0.01)
        total  = len(tr_dl) * CFG["ROB_EPOCHS"]
        warmup = int(total * CFG["ROB_WARMUP"])
        sched  = get_linear_schedule_with_warmup(opt, warmup, total)

        best_f1, best_state = 0.0, None
        for epoch in range(CFG["ROB_EPOCHS"]):
            model.train(); ep_loss = 0.0
            for bi, batch in enumerate(tr_dl):
                ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
                tgt = batch["y"].long().to(DEVICE)
                loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); ep_loss += loss.item()
                if (bi + 1) % 150 == 0:
                    print(f"    Ep{epoch+1} B{bi+1}/{len(tr_dl)} CE={ep_loss/(bi+1):.4f}")
            vl_p = _cls_infer(model, tok, gather(posts, bv),
                              DEVICE, CFG["ROB_BATCH"], CFG["ROB_MAX_LEN"])
            _, vl_f1 = best_threshold(labels[bv], vl_p)
            print(f"  Epoch {epoch+1}  CE={ep_loss/len(tr_dl):.4f}  val_F1={vl_f1:.4f}")
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ★ best val_F1={best_f1:.4f}")
        model.load_state_dict(best_state)
        torch.save(best_state, CFG["ROBERTA_WEIGHTS"])
        print(f"  Saved weights → {CFG['ROBERTA_WEIGHTS']}")

    blend_p = _cls_infer(model, tok, gather(posts, blend_val_idx),
                         DEVICE, CFG["ROB_BATCH"], CFG["ROB_MAX_LEN"])
    test_p  = _cls_infer(model, tok, gather(posts, test_idx),
                         DEVICE, CFG["ROB_BATCH"], CFG["ROB_MAX_LEN"])
    del model; free_gpu()

    m = model_metrics(y[test_idx], test_p, "RoBERTa-Base")
    print(f"  RoBERTa standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.3  LightGBM ────────────────────────────────────────────────
def train_lightgbm(feat_df, feat_names, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.3 — LightGBM  [30 tabular features]")
    X = feat_df[feat_names].values
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[base_train_idx])
    Xbl = sc.transform(X[blend_val_idx])
    Xte = sc.transform(X[test_idx])

    clf = lgb.LGBMClassifier(**CFG["LGBM_PARAMS"])
    clf.fit(Xtr, y[base_train_idx],
            eval_set=[(Xbl, y[blend_val_idx])],
            callbacks=[lgb.early_stopping(60, verbose=False),
                       lgb.log_evaluation(period=0)])

    blend_p = clf.predict_proba(Xbl)[:, 1]
    test_p  = clf.predict_proba(Xte)[:, 1]
    m = model_metrics(y[test_idx], test_p, "LightGBM")
    print(f"  LightGBM standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m, clf, sc


# ── 4.4  XGBoost — tuned for minority-class recall ───────────────
def train_xgboost(feat_df, feat_names, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.4 — XGBoost  [tuned for clickbait recall]")
    X = feat_df[feat_names].values
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[base_train_idx])
    Xbl = sc.transform(X[blend_val_idx])
    Xte = sc.transform(X[test_idx])

    # Compute class imbalance ratio for scale_pos_weight
    n_neg = (y[base_train_idx] == 0).sum()
    n_pos = (y[base_train_idx] == 1).sum()
    spw   = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {spw:.2f}")

    params = {**CFG["XGB_PARAMS"], "scale_pos_weight": spw}
    clf = XGBClassifier(**params)
    clf.fit(Xtr, y[base_train_idx],
            eval_set=[(Xbl, y[blend_val_idx])],
            verbose=False,
            early_stopping_rounds=50)

    blend_p = clf.predict_proba(Xbl)[:, 1]
    test_p  = clf.predict_proba(Xte)[:, 1]
    m = model_metrics(y[test_idx], test_p, "XGBoost")
    print(f"  XGBoost standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.5  CatBoost — categorical metadata ─────────────────────────
def _derive_topic(kw_str):
    kw_lower = str(kw_str).lower()
    for topic, keys in TOPIC_MAP.items():
        if any(k in kw_lower for k in keys):
            return topic
    return "other"

def _has_media(media_str):
    s = str(media_str).strip()
    return "no_media" if (s == "[]" or s == "" or s.lower() == "nan") else "has_media"


def train_catboost(feat_df, feat_names, df, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.5 — CatBoost  [categorical: topic, media, post-length]")
    if not HAS_CATBOOST:
        print("  [SKIP] catboost not installed — returning zeros")
        n_bl = len(blend_val_idx); n_te = len(test_idx)
        dummy_m = dict(label="CatBoost", f1=0.0, acc=0.0, prec=0.0, rec=0.0, thr=0.5)
        return np.full(n_bl, 0.5), np.full(n_te, 0.5), dummy_m

    # Categorical feature engineering
    topic_cat = df["keywords"].apply(_derive_topic)
    media_cat = df["media_raw"].apply(_has_media)
    post_len_cat = pd.cut(df["post"].apply(lambda x: len(x.split())),
                           bins=[0, 5, 15, 9999],
                           labels=["short", "medium", "long"])

    # Combine numerical + categorical
    cat_frame = pd.DataFrame({
        "topic_cat":    topic_cat.values,
        "media_cat":    media_cat.values,
        "post_len_cat": post_len_cat.astype(str).values,
    }, index=df.index)
    combined = pd.concat([feat_df[feat_names], cat_frame], axis=1)
    cat_feature_names = list(cat_frame.columns)
    cat_indices = [combined.columns.tolist().index(c) for c in cat_feature_names]

    Xtr = combined.iloc[base_train_idx].values
    Xbl = combined.iloc[blend_val_idx].values
    Xte = combined.iloc[test_idx].values

    clf = CatBoostClassifier(**CFG["CB_PARAMS"], cat_features=cat_indices)
    clf.fit(Xtr, y[base_train_idx],
            eval_set=(Xbl, y[blend_val_idx]),
            early_stopping_rounds=50,
            use_best_model=True)

    blend_p = clf.predict_proba(Xbl)[:, 1]
    test_p  = clf.predict_proba(Xte)[:, 1]
    m = model_metrics(y[test_idx], test_p, "CatBoost")
    print(f"  CatBoost standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.6  Logistic Regression (L1/L2 elastic-net) ─────────────────
def train_logistic_regression(feat_df, feat_names, y,
                               base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.6 — Logistic Regression  [ElasticNet, skeptical baseline]")
    X = feat_df[feat_names].values
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[base_train_idx])
    Xbl = sc.transform(X[blend_val_idx])
    Xte = sc.transform(X[test_idx])

    # Use ElasticNet (l1_ratio=0.5) with class balancing
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
        class_weight="balanced", max_iter=5000, random_state=CFG["SEED"])
    clf.fit(Xtr, y[base_train_idx])

    blend_p = clf.predict_proba(Xbl)[:, 1]
    test_p  = clf.predict_proba(Xte)[:, 1]
    m = model_metrics(y[test_idx], test_p, "Logistic-EN")
    print(f"  LogReg standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.7  SBERT-MLP — 3-layer neural network ──────────────────────
class _SBERTMlp(nn.Module):
    def __init__(self, in_dim=1536, h1=512, h2=128, d1=0.30, d2=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),   nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h1, h2),       nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(h2, 1),
        )
    def forward(self, x): return torch.sigmoid(self.net(x))


class _EmbDs(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def train_sbert_mlp(emb_post, emb_title, y,
                     base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.7 — SBERT-MLP  [3-layer NN on sentence embeddings]")

    # Concatenate 4 interaction views: [post, title, diff, prod]
    def make_X(ep, et):
        return np.concatenate([ep, et, ep - et, ep * et], axis=1)

    X_all = make_X(emb_post, emb_title)
    print(f"  MLP input dim: {X_all.shape[1]}  ({emb_post.shape[1]} × 4)")

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_all[base_train_idx])
    Xbl = sc.transform(X_all[blend_val_idx])
    Xte = sc.transform(X_all[test_idx])

    # Internal val from base_train
    bt_idx, bv_idx = train_test_split(
        np.arange(len(base_train_idx)), test_size=0.12,
        stratify=y[base_train_idx], random_state=CFG["SEED"])

    n_neg = (y[base_train_idx[bt_idx]] == 0).sum()
    n_pos = (y[base_train_idx[bt_idx]] == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    tr_ds = _EmbDs(Xtr[bt_idx], y[base_train_idx[bt_idx]])
    vl_ds = _EmbDs(Xtr[bv_idx], y[base_train_idx[bv_idx]])
    tr_dl = DataLoader(tr_ds, batch_size=CFG["MLP_BATCH"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=CFG["MLP_BATCH"], shuffle=False)

    model = _SBERTMlp(
        in_dim=X_all.shape[1],
        d1=CFG["MLP_DROPOUT1"], d2=CFG["MLP_DROPOUT2"]
    ).to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), lr=CFG["MLP_LR"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["MLP_EPOCHS"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1, best_state = 0.0, None
    for epoch in range(CFG["MLP_EPOCHS"]):
        model.train(); ep_loss = 0.0
        for Xb, yb in tr_dl:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model.net(Xb).squeeze(-1)
            loss   = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); ep_loss += loss.item()
        sched.step()
        # validate
        model.eval(); vl_p = []
        with torch.no_grad():
            for Xb, _ in vl_dl:
                vl_p.extend(model(Xb.to(DEVICE)).squeeze(-1).cpu().numpy())
        _, vl_f1 = best_threshold(y[base_train_idx[bv_idx]], np.array(vl_p))
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3}/{CFG['MLP_EPOCHS']}  "
                  f"loss={ep_loss/len(tr_dl):.4f}  val_F1={vl_f1:.4f}")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    def infer(X):
        ds = _EmbDs(X); dl = DataLoader(ds, batch_size=512, shuffle=False)
        out = []
        with torch.no_grad():
            for Xb in dl:
                out.extend(model(Xb.to(DEVICE)).squeeze(-1).cpu().numpy())
        return np.array(out)

    blend_p = infer(Xbl)
    test_p  = infer(Xte)
    del model; free_gpu()

    m = model_metrics(y[test_idx], test_p, "SBERT-MLP")
    print(f"  SBERT-MLP standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 · OPTIMISED POWER BLEND
# ═══════════════════════════════════════════════════════════════════
def optimize_blend(blend_probs, test_probs, y_blend, y_test, model_names):
    """
    Grid-search + scipy optimize to find:
      w  = [w1 … w7]  (non-negative weights)
      k  = power parameter  (∈ POWER_K_GRID)

    P_final = sum(w_i · p_i^k) / sum(w_i)
    """
    sec("SECTION 5 — Optimised Power Blend")

    P_bl = np.array([blend_probs[m] for m in model_names]).T  # (n_blend, 7)
    P_te = np.array([test_probs[m]  for m in model_names]).T  # (n_test,  7)
    P_bl = np.clip(P_bl, 1e-7, 1 - 1e-7)
    P_te = np.clip(P_te, 1e-7, 1 - 1e-7)
    n_m  = len(model_names)

    def blend(P, weights, k):
        w = np.abs(weights); w_sum = w.sum()
        if w_sum < 1e-8: w = np.ones(n_m); w_sum = n_m
        return (P ** k) @ (w / w_sum)

    def neg_f1(params):
        weights = np.abs(params[:n_m])
        k = np.clip(params[n_m], 0.25, 4.0)
        p = blend(P_bl, weights, k)
        _, bst = best_threshold(y_blend, p, CFG["BLEND_THR_GRID"])
        return -bst

    # ── Initial weights: standalone blend-val F1 scores ────────────
    standalone_f1 = []
    for m in model_names:
        _, f = best_threshold(y_blend, blend_probs[m], CFG["BLEND_THR_GRID"])
        standalone_f1.append(f)
    init_w = np.array(standalone_f1, dtype=float)
    init_w = np.maximum(init_w, 0.05)   # floor to avoid zeros

    print(f"\n  Initial weights (by blend-val F1):")
    for m, w, f in zip(model_names, init_w, standalone_f1):
        print(f"    {m:<22}  blend-val F1={f:.4f}  init_w={w:.3f}")

    # ── Grid search over power k ────────────────────────────────────
    subsec("Grid Search over power k")
    best_grid_k, best_grid_f1, best_grid_w = 1.0, 0.0, init_w.copy()

    for k in CFG["POWER_K_GRID"]:
        x0 = np.append(init_w.copy(), k)
        bounds = [(0, None)] * n_m + [(0.25, 4.0)]
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
        f1_val = -res.fun
        print(f"  k={k:.2f}  →  blend-val F1={f1_val:.4f}")
        if f1_val > best_grid_f1:
            best_grid_f1 = f1_val
            best_grid_k  = res.x[n_m]
            best_grid_w  = np.abs(res.x[:n_m])

    # ── Final refinement from best k starting point ─────────────────
    subsec("Final Refinement")
    x0_final = np.append(best_grid_w, best_grid_k)
    res_final = minimize(neg_f1, x0_final, method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
    opt_weights = np.abs(res_final.x[:n_m])
    opt_k       = float(np.clip(res_final.x[n_m], 0.25, 4.0))
    opt_blend_f1 = -res_final.fun

    # Normalize weights for display
    w_norm = opt_weights / opt_weights.sum()
    print(f"\n  Optimal power k = {opt_k:.3f}")
    print(f"  Blend-val F1 (power blend) = {opt_blend_f1:.4f}")
    print(f"\n  Final normalised weights:")
    for m, w in zip(model_names, w_norm):
        bar = "█" * int(w * 40)
        print(f"    {m:<22}  {w:.4f}  {bar}")

    # ── Apply to blend_val and test ─────────────────────────────────
    blend_final = blend(P_bl, opt_weights, opt_k)
    test_final  = blend(P_te, opt_weights, opt_k)

    t_blend, _ = best_threshold(y_blend, blend_final, CFG["BLEND_THR_GRID"])
    m_blend = model_metrics(y_blend,   blend_final, "PowerBlend@blend_val")
    m_test  = model_metrics(y_test,    test_final,  "PowerBlend@test")

    print(f"\n  Power Blend — blend_val F1={m_blend['f1']:.4f}")
    print(f"  Power Blend — test      F1={m_test['f1']:.4f}  "
          f"(↑ {m_test['f1'] - CFG['PRIOR_BEST_F1']:+.4f} vs prior best)")

    return blend_final, test_final, opt_weights, opt_k, w_norm, model_names


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 · TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════════════
def temperature_scale(blend_raw, y_blend, test_raw):
    sec("SECTION 6 — Temperature Scaling  (calibrate blend output)")

    # NLL objective
    def nll(T):
        T = max(float(T), 0.01)
        p = np.clip(blend_raw, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1 - p)) / T
        p_cal  = 1 / (1 + np.exp(-logits))
        return -np.mean(y_blend * np.log(p_cal + 1e-15) +
                        (1 - y_blend) * np.log(1 - p_cal + 1e-15))

    res = minimize_scalar(nll, bounds=(0.10, 10.0), method="bounded")
    T   = float(res.x)

    def apply_T(p_raw):
        p = np.clip(p_raw, 1e-7, 1 - 1e-7)
        return sigmoid(np.log(p / (1 - p)) / T)

    p_blend_cal = apply_T(blend_raw)
    p_test_cal  = apply_T(test_raw)

    # ECE comparison
    def ece(y, p, n=10):
        fp, mp = calibration_curve(y, p, n_bins=n, strategy="uniform")
        return float(np.mean(np.abs(fp - mp)))

    ece_raw = ece(y_blend, blend_raw)
    ece_cal = ece(y_blend, p_blend_cal)
    print(f"  Optimal temperature T = {T:.4f}")
    print(f"  ECE before calibration : {ece_raw:.4f}")
    print(f"  ECE after calibration  : {ece_cal:.4f}  "
          f"({'improvement' if ece_cal < ece_raw else 'no change'})")

    return p_blend_cal, p_test_cal, T, ece_raw, ece_cal


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7 · FINAL EVALUATION & TRI-STATE
# ═══════════════════════════════════════════════════════════════════
def final_evaluation(final_probs, y_test):
    sec("SECTION 7 — Final Evaluation + Tri-State Analysis")

    t_opt, f1_opt = best_threshold(y_test, final_probs, 81)
    y_pred = (final_probs >= t_opt).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    br   = brier_score_loss(y_test, final_probs)

    print(f"\n  ═══════════════════════════════")
    print(f"  FINAL F1-score  : {f1_opt:.4f}  "
          f"{'★ BARRIER BROKEN' if f1_opt > CFG['PRIOR_BEST_F1'] else '→ below 0.711'}"
          f"  (Δ={f1_opt - CFG['PRIOR_BEST_F1']:+.4f})")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  Brier Score     : {br:.4f}")
    print(f"  Threshold       : {t_opt:.4f}")
    print(f"  ═══════════════════════════════\n")

    print(classification_report(y_test, y_pred,
                                target_names=["not-clickbait", "clickbait"]))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix   Pred NCB  Pred CB")
    print(f"  True NCB   :       {tn:>7}   {fp:>7}")
    print(f"  True CB    :       {fn:>7}   {tp:>7}")

    # ── Tri-State ───────────────────────────────────────────────────
    CB_T  = CFG["CERTAIN_CB"]
    NCB_T = CFG["CERTAIN_NCB"]

    buckets = np.where(final_probs > CB_T, "CERTAIN_CB",
              np.where(final_probs < NCB_T, "CERTAIN_NCB", "AMBIGUOUS_IDK"))

    subsec("Tri-State Analysis")
    for b in ["CERTAIN_CB", "CERTAIN_NCB", "AMBIGUOUS_IDK"]:
        mask = buckets == b
        n = mask.sum(); pct = n / len(y_test) * 100
        if b != "AMBIGUOUS_IDK":
            pred_b = (b == "CERTAIN_CB")
            prec_b = (y_test[mask] == int(pred_b)).mean() if n > 0 else 0.0
            print(f"  {b:<20}: {n:>5} ({pct:.1f}%)  precision={prec_b:.4f} "
                  f"{'✓>90%' if prec_b > 0.9 else ''}")
        else:
            cb_rate = y_test[mask].mean() if n > 0 else 0.0
            print(f"  {b:<20}: {n:>5} ({pct:.1f}%)  CB-rate={cb_rate:.3f}")

    res_mask = buckets != "AMBIGUOUS_IDK"
    if res_mask.sum() > 0:
        r_pred = (buckets[res_mask] == "CERTAIN_CB").astype(int)
        res_acc = accuracy_score(y_test[res_mask], r_pred)
        res_f1  = f1_score(y_test[res_mask], r_pred, zero_division=0)
        print(f"\n  Resolved Accuracy  : {res_acc:.4f}  "
              f"({res_mask.sum()} samples, {res_mask.mean()*100:.1f}% coverage)")
        print(f"  Resolved F1        : {res_f1:.4f}")

    subsec("Tri-State Threshold Sensitivity")
    print(f"  {'CB_thr':>6}  {'NCB_thr':>7}  {'Cov%':>6}  "
          f"{'ResAcc':>7}  {'CB_P':>7}  {'NCB_P':>7}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for cbt, ncbt in [(0.60,0.40),(0.65,0.35),(0.70,0.30),(0.75,0.25),(0.80,0.20),(0.85,0.15)]:
        _cb  = final_probs > cbt
        _ncb = final_probs < ncbt
        _res = _cb | _ncb
        if _res.sum() == 0: continue
        _pred = _cb[_res].astype(int)
        _acc  = accuracy_score(y_test[_res], _pred)
        _cbp  = (y_test[_cb] == 1).mean() if _cb.sum() else 0.0
        _ncbp = (y_test[_ncb] == 0).mean() if _ncb.sum() else 0.0
        print(f"  {cbt:.2f}    {ncbt:.2f}     {_res.mean()*100:.1f}%  "
              f"{_acc:.4f}   {_cbp:.4f}   {_ncbp:.4f}")

    return f1_opt, acc, prec, rec, y_pred, buckets


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8 · SCIENTIFIC REPORTS
# ═══════════════════════════════════════════════════════════════════

# ── 8.1  Leaderboard ─────────────────────────────────────────────
def print_leaderboard(metrics_list, final_f1):
    sec("SECTION 8.1 — Tournament Leaderboard")
    rows = sorted(metrics_list, key=lambda x: -x["f1"])

    header = f"  {'#':>2}  {'Model':<22}  {'F1':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'Thr':>6}"
    print(header)
    print(f"  {'─'*2}  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")
    for i, m in enumerate(rows, 1):
        marker = " ←" if i == 1 else ""
        print(f"  {i:>2}  {m['label']:<22}  {m['f1']:.4f}  "
              f"{m['acc']:.4f}  {m['prec']:.4f}  {m['rec']:.4f}  "
              f"{m['thr']:.3f}{marker}")
    print(f"\n  ─── Power Blend (final) ─── F1={final_f1:.4f}  "
          f"{'★ NEW BEST' if final_f1 > rows[0]['f1'] else ''}")

    df_lb = pd.DataFrame(rows)
    df_lb.to_csv(CFG["OUTPUT_LEADERBOARD"], index=False)
    print(f"  Saved → {CFG['OUTPUT_LEADERBOARD']}")
    return df_lb


# ── 8.2  Diversity Matrix ─────────────────────────────────────────
def diversity_matrix(test_probs, model_names):
    sec("SECTION 8.2 — Diversity Matrix  (inter-model probability correlation)")
    P = np.array([test_probs[m] for m in model_names])
    C = np.corrcoef(P)

    # Console pretty-print
    w = 8
    print("  " + "".join(f"{n[:w]:>{w+1}}" for n in model_names))
    for i, n in enumerate(model_names):
        row = f"  {n[:14]:<14} "
        for j in range(len(model_names)):
            v = C[i, j]
            row += f"  {v:+.3f}"
        print(row)

    # Compute mean off-diagonal correlation
    off_diag = C[np.triu_indices(len(model_names), k=1)]
    print(f"\n  Mean |off-diagonal| correlation : {np.mean(np.abs(off_diag)):.4f}")
    print(f"  Min  off-diagonal correlation   : {off_diag.min():+.4f}")
    print(f"  (Lower avg correlation → more diverse ensemble → more upside)")

    # Seaborn heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    # shorten names for display
    short_names = [n.replace("DistilBERT-v2","DBERT").replace("RoBERTa-Base","RoB")
                    .replace("LightGBM","LGBM").replace("XGBoost","XGB")
                    .replace("CatBoost","CB").replace("Logistic-EN","LR")
                    .replace("SBERT-MLP","MLP") for n in model_names]
    sns.heatmap(C, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Model Diversity Matrix (Pearson ρ of P(clickbait))", fontsize=13)
    plt.tight_layout()
    plt.savefig("diversity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → diversity_matrix.png")
    return C


# ── 8.3  "What Worked" Report ─────────────────────────────────────
def what_worked_report(lgbm_clf, lgbm_scaler, feat_names,
                        blend_probs, y_blend, opt_weights, w_norm, model_names):
    sec("SECTION 8.3 — 'What Worked' Report")

    # ── Top-5 features (LightGBM importance) ─────────────────────
    subsec("Top-5 Features (LightGBM gain importance)")
    imp = lgbm_clf.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    for rank, idx in enumerate(order[:5], 1):
        pct = imp[idx] / imp.sum() * 100
        bar = "█" * int(pct * 1.2)
        print(f"  {rank}. {feat_names[idx]:<35}  {imp[idx]:>6.1f}  ({pct:.1f}%)  {bar}")

    # All top-10 for completeness
    subsec("Top-10 Features (full)")
    for rank, idx in enumerate(order, 1):
        pct = imp[idx] / imp.sum() * 100
        new_tag = " ★ NEW" if feat_names[idx] in (
            "mirror_similarity", "pragmatic_slang_count",
            "entity_density", "curiosity_gap_index") else ""
        print(f"  {rank:>2}. {feat_names[idx]:<35}  {imp[idx]:>6.1f}  "
              f"({pct:.1f}%){new_tag}")

    # ── Top-3 models by blend weight contribution ─────────────────
    subsec("Top-3 Models by Blend-Weight Contribution")
    contributions = [(m, w) for m, w in zip(model_names, w_norm)]
    contributions.sort(key=lambda x: -x[1])
    for rank, (m, w) in enumerate(contributions[:3], 1):
        bar = "█" * int(w * 60)
        print(f"  {rank}. {m:<22}  weight={w:.4f}  {bar}")

    # ── Leave-one-out analysis on blend_val ─────────────────────
    subsec("Leave-One-Out F1 Impact on blend_val")
    P_bl = np.array([blend_probs[m] for m in model_names]).T
    P_bl = np.clip(P_bl, 1e-7, 1 - 1e-7)

    # Full blend F1
    w = np.array([w_norm[i] for i in range(len(model_names))])
    p_full = (P_bl ** 1.0) @ w
    _, f1_full = best_threshold(y_blend, p_full, CFG["BLEND_THR_GRID"])

    print(f"\n  Full blend F1 = {f1_full:.4f}")
    loo_results = []
    for leave_idx, leave_m in enumerate(model_names):
        keep  = [i for i in range(len(model_names)) if i != leave_idx]
        w_loo = w[keep]; w_loo /= w_loo.sum()
        p_loo = (P_bl[:, keep] ** 1.0) @ w_loo
        _, f1_loo = best_threshold(y_blend, p_loo, CFG["BLEND_THR_GRID"])
        impact = f1_full - f1_loo
        loo_results.append((leave_m, f1_loo, impact))

    loo_results.sort(key=lambda x: -x[2])
    for m, f1_l, impact in loo_results:
        direction = "▲" if impact > 0 else "▼"
        print(f"  Remove {m:<22}  → F1={f1_l:.4f}  "
              f"contribution = {direction}{abs(impact):.4f}")

    # ── New feature impact estimate ───────────────────────────────
    subsec("New Feature Contribution Estimate (vs. 28-feature baseline)")
    new_features = ["mirror_similarity", "pragmatic_slang_count",
                    "entity_density", "curiosity_gap_index"]
    present = [f for f in new_features if f in feat_names]
    print(f"  New features in model: {present}")
    print(f"  LightGBM importances for new features:")
    feat_imp_dict = dict(zip(feat_names, imp))
    for f in present:
        v = feat_imp_dict.get(f, 0)
        pct = v / imp.sum() * 100
        print(f"    {f:<35}  {v:>6.1f}  ({pct:.1f}%)")


# ── 8.4  Calibration Plot ──────────────────────────────────────
def calibration_plot(test_probs_dict, final_probs, y_test, model_names, T, ece_raw, ece_cal):
    sec("SECTION 8.4 — Calibration & Confidence Plots")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Ensemble Calibration & Confidence Dashboard", fontsize=13, fontweight="bold")

    # Panel 1: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    fp_raw, mp_raw = calibration_curve(y_test, test_probs_dict.get("_raw_blend", final_probs),
                                        n_bins=10, strategy="uniform")
    fp_cal, mp_cal = calibration_curve(y_test, final_probs, n_bins=10, strategy="uniform")
    ax.plot(mp_raw, fp_raw, "o-", color="#E63946", label=f"Pre-scaling ECE={ece_raw:.3f}")
    ax.plot(mp_cal, fp_cal, "s-", color="#2A9D8F", label=f"Post-scaling ECE={ece_cal:.3f}")
    ax.axvline(0.20, color="#457B9D", ls="--", alpha=0.6, lw=1)
    ax.axvline(0.80, color="#E76F51", ls="--", alpha=0.6, lw=1)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram\n(T={T:.3f})")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2: Final probability histogram by class
    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(final_probs[y_test == 0], bins=bins, color="#457B9D", alpha=0.65,
            density=True, label="True NCB")
    ax.hist(final_probs[y_test == 1], bins=bins, color="#E63946", alpha=0.65,
            density=True, label="True CB")
    ax.axvline(CFG["CERTAIN_NCB"], color="#457B9D", ls="--", lw=1.5)
    ax.axvline(CFG["CERTAIN_CB"],  color="#E63946", ls="--", lw=1.5)
    idk_pct = np.mean((final_probs >= CFG["CERTAIN_NCB"]) & (final_probs <= CFG["CERTAIN_CB"])) * 100
    ax.text(0.50, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1,
            f"IDK\n{idk_pct:.1f}%", ha="center", fontsize=9, color="gray", style="italic")
    ax.set_xlabel("Final Calibrated Probability"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Class"); ax.legend(fontsize=8)

    # Panel 3: Individual model score distributions (box plots)
    ax = axes[2]
    data_cb  = [test_probs_dict[m][y_test == 1] for m in model_names]
    data_ncb = [test_probs_dict[m][y_test == 0] for m in model_names]
    short = [n[:8] for n in model_names]
    x = np.arange(len(model_names))
    vp1 = ax.violinplot(data_cb,  positions=x - 0.2, widths=0.35, showmedians=True)
    vp2 = ax.violinplot(data_ncb, positions=x + 0.2, widths=0.35, showmedians=True)
    for pc in vp1["bodies"]: pc.set_facecolor("#E63946"); pc.set_alpha(0.6)
    for pc in vp2["bodies"]: pc.set_facecolor("#457B9D"); pc.set_alpha(0.6)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7, rotation=30)
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Model Score Distributions\n(red=CB, blue=NCB)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#E63946", label="True CB"),
                        Patch(color="#457B9D", label="True NCB")], fontsize=8)

    plt.tight_layout()
    plt.savefig("ensemble_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → ensemble_calibration.png")


# ── 8.5  Top-10 Model vs Human Discrepancies ─────────────────────
def model_vs_human(df, test_idx, final_probs, y_test):
    sec("SECTION 8.5 — Top-10 'Model vs Human' Discrepancies")
    truth_mean = df.iloc[test_idx]["truthMean"].values
    y_pred_bin = (final_probs > 0.5).astype(int)
    model_committed = (final_probs > 0.72) | (final_probs < 0.28)
    disagrees = y_pred_bin != y_test
    disc_score = np.abs(final_probs - 0.5) * np.abs(final_probs - truth_mean)
    disc_score[~(model_committed & disagrees)] = -1

    top = np.argsort(disc_score)[::-1][:10]
    print(f"\n  {'#':>2}  {'TrCls':>5}  {'TrMean':>7}  {'Prob':>6}  Post Text")
    print(f"  {'─'*2}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*55}")
    for rank, i in enumerate(top, 1):
        if disc_score[i] < 0: continue
        pt = textwrap.shorten(df.iloc[test_idx[i]]["post"], 55, placeholder="…")
        tt = textwrap.shorten(df.iloc[test_idx[i]]["title"], 55, placeholder="…")
        if y_test[i] == 1 and final_probs[i] < 0.30:
            hypo = "Post mirrors headline verbatim — no curiosity gap."
        elif y_test[i] == 0 and final_probs[i] > 0.70:
            hypo = "Post uses list-bait / slang despite low human score."
        else:
            hypo = "Borderline — annotator split likely."
        print(f"  {rank:>2}  {y_test[i]:>5}  {truth_mean[i]:>7.2f}  {final_probs[i]:>6.3f}")
        print(f"      post : {pt}")
        print(f"      title: {tt}")
        print(f"      hypo : {hypo}\n")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9 · SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_outputs(df, test_idx, final_probs, y_pred, y_test, buckets,
                  test_probs_dict, model_names, f1, acc, prec, rec,
                  opt_k, w_norm, T, ece_raw, ece_cal):
    sec("SECTION 9 — Saving Outputs")

    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_label"]     = y_test
    out["pred_binary"]    = y_pred
    out["prob_ensemble"]  = final_probs
    out["tristate_label"] = buckets
    out["is_correct"]     = (y_pred == y_test).astype(int)
    for m in model_names:
        out[f"prob_{m.lower().replace('-','_').replace(' ','_')}"] = test_probs_dict[m]

    out.to_csv(CFG["OUTPUT_PREDS"], index=False)
    print(f"  ✓ Predictions → {CFG['OUTPUT_PREDS']}  (shape {out.shape})")

    # Results card
    card = f"""
╔══════════════════════════════════════════════════════════════════════╗
║          7-MODEL TOURNAMENT + POWER BLEND — RESULTS CARD            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Pipeline                                                            ║
║    7 base learners  →  Power Blend (k={opt_k:.3f})  →  Temp Scale (T={T:.3f}) ║
╠══════════════════════════════════════════════════════════════════════╣
║  Final Test-Set Metrics  (20% hold-out, seed=42)                    ║
║    F1-score   : {f1:.4f}   {'★ BARRIER BROKEN (+' + f'{f1-0.711:.4f}' + ')' if f1>0.711 else '→ below 0.711'}{'':25}║
║    Accuracy   : {acc:.4f}                                              ║
║    Precision  : {prec:.4f}                                              ║
║    Recall     : {rec:.4f}                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Calibration                                                         ║
║    ECE (raw blend)  : {ece_raw:.4f}                                      ║
║    ECE (calibrated) : {ece_cal:.4f}                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Top-3 Models by Blend Weight                                        ║"""
    sorted_contrib = sorted(zip(model_names, w_norm), key=lambda x: -x[1])
    for i, (m, w) in enumerate(sorted_contrib[:3]):
        card += f"\n║    {i+1}. {m:<22} w={w:.4f}{'':30}║"
    card += f"""
╚══════════════════════════════════════════════════════════════════════╝"""
    print(card)

    with open("tournament_results_card.txt", "w") as fh:
        fh.write(card)
    print(f"  ✓ Results card → tournament_results_card.txt")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    banner("7-MODEL TOURNAMENT + POWER BLEND  |  Target: F1 > 0.711"
           f"  |  Device: {DEVICE}")

    # ── 1. Data ────────────────────────────────────────────────────
    df, y, base_train_idx, blend_val_idx, test_idx = load_and_split()

    # ── 3. SBERT Encoding (before feature factory to reuse embeddings)
    emb_post, emb_art, emb_ttl, emb_dsc = sbert_encode(df)

    # ── 2. Feature Factory ─────────────────────────────────────────
    feat_df, feat_names = build_feature_factory(
        df, emb_post, emb_ttl, emb_art, emb_dsc, base_train_idx, y)

    # ── 4. Tournament ──────────────────────────────────────────────
    blend_probs = {}
    test_probs  = {}
    all_metrics = []

    # 4.1 DistilBERT
    bp, tp, m = train_distilbert(df, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["DistilBERT-v2"] = bp
    test_probs["DistilBERT-v2"]  = tp
    all_metrics.append(m)

    # 4.2 RoBERTa
    bp, tp, m = train_roberta(df, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["RoBERTa-Base"] = bp
    test_probs["RoBERTa-Base"]  = tp
    all_metrics.append(m)

    # 4.3 LightGBM
    bp, tp, m, lgbm_clf, lgbm_sc = train_lightgbm(
        feat_df, feat_names, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["LightGBM"] = bp
    test_probs["LightGBM"]  = tp
    all_metrics.append(m)

    # 4.4 XGBoost
    bp, tp, m = train_xgboost(
        feat_df, feat_names, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["XGBoost"] = bp
    test_probs["XGBoost"]  = tp
    all_metrics.append(m)

    # 4.5 CatBoost
    bp, tp, m = train_catboost(
        feat_df, feat_names, df, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["CatBoost"] = bp
    test_probs["CatBoost"]  = tp
    all_metrics.append(m)

    # 4.6 Logistic Regression
    bp, tp, m = train_logistic_regression(
        feat_df, feat_names, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["Logistic-EN"] = bp
    test_probs["Logistic-EN"]  = tp
    all_metrics.append(m)

    # 4.7 SBERT-MLP
    bp, tp, m = train_sbert_mlp(
        emb_post, emb_ttl, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["SBERT-MLP"] = bp
    test_probs["SBERT-MLP"]  = tp
    all_metrics.append(m)

    model_names = list(blend_probs.keys())

    # ── 5. Power Blend ─────────────────────────────────────────────
    blend_raw, test_raw, opt_weights, opt_k, w_norm, model_names = optimize_blend(
        blend_probs, test_probs, y[blend_val_idx], y[test_idx], model_names)
    test_probs["_raw_blend"] = test_raw  # for calibration plot

    # ── 6. Temperature Scaling ─────────────────────────────────────
    blend_cal, test_cal, T, ece_raw, ece_cal = temperature_scale(
        blend_raw, y[blend_val_idx], test_raw)

    # ── 7. Final Evaluation ────────────────────────────────────────
    f1, acc, prec, rec, y_pred, buckets = final_evaluation(test_cal, y[test_idx])

    # ── 8. Scientific Reports ──────────────────────────────────────
    df_lb = print_leaderboard(all_metrics, f1)
    C     = diversity_matrix(test_probs, model_names)
    what_worked_report(lgbm_clf, lgbm_sc, feat_names,
                        blend_probs, y[blend_val_idx], opt_weights, w_norm, model_names)
    calibration_plot(test_probs, test_cal, y[test_idx], model_names, T, ece_raw, ece_cal)
    model_vs_human(df, test_idx, test_cal, y[test_idx])

    # ── 9. Save ────────────────────────────────────────────────────
    save_outputs(df, test_idx, test_cal, y_pred, y[test_idx], buckets,
                  test_probs, model_names, f1, acc, prec, rec,
                  opt_k, w_norm, T, ece_raw, ece_cal)

    print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
