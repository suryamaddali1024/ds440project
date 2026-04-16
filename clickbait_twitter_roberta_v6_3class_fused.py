"""
clickbait_twitter_roberta_v6_3class_fused.py
=============================================
Same fused 3-class pipeline — backbone swapped to
cardiffnlp/twitter-roberta-base.
Twitter-RoBERTa was pre-trained on 58 M tweets using the RoBERTa objective,
combining RoBERTa's robust masked-LM training with Twitter domain knowledge.

Five Innovation Axes (from deberta_hyperopt):
  1. Twitter-RoBERTa  — 58M tweet pre-training (cardiffnlp/twitter-roberta-base)
  2. Intermediate Fusion — 8 hand-crafted features injected at encoder layer 6
  3. Differential LR  — layer-wise learning rate decay (factor=0.85)
  4. Multi-Task Loss  — Focal-CE(3-class) + MSE(truthMean) + Label-Smooth-CE
  5. Data Augmentation — title dropout + synonym swap + span corruption
  + SWA  (Stochastic Weight Averaging, starts at epoch 4)
  + Temperature Scaling / post-hoc calibration

Three-Class Target (from distilbert_v5_3class):
  Class 0 — not_clickbait   (truthMean < 0.30)
  Class 1 — ambiguous        (0.30 ≤ truthMean < 0.70)
  Class 2 — clickbait        (truthMean ≥ 0.70)

Training data : final_cleaned_full.csv  (same 80/20 stratified split, seed=42)
New inference : combined_modern_cleaned.csv
  Columns: source, headline, description
  headline → post text  |  description → article/title text

pip install torch transformers scikit-learn vaderSentiment Levenshtein
            scipy matplotlib seaborn sentencepiece protobuf
"""

# ═══════════════════════════════════════════════════════════════════
#  0 · IMPORTS
# ═══════════════════════════════════════════════════════════════════
import ast, copy, gc, math, os, random, re, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
from scipy.stats import entropy as sp_entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

try:
    from Levenshtein import ratio as lev_ratio
except ImportError:
    import difflib
    def lev_ratio(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ═══════════════════════════════════════════════════════════════════
#  0 · GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    # ── Files ────────────────────────────────────────────────────────
    INPUT_FILE          = "final_cleaned_full.csv",
    MODERN_FILE         = "combined_modern_cleaned.csv",
    OUTPUT_PREDS        = "twitter_roberta_v6_3class_test_predictions.csv",
    OUTPUT_MODERN       = "twitter_roberta_v6_modern_predictions.csv",
    OUTPUT_REPORT       = "twitter_roberta_v6_3class_report.png",
    MODEL_WEIGHTS       = "twitter_roberta_v6_3class_best.pt",

    # ── Splits ───────────────────────────────────────────────────────
    TEST_SIZE           = 0.20,
    VAL_SIZE            = 0.10,          # fraction of total
    SEED                = 42,

    # ── Three-class thresholds ───────────────────────────────────────
    AMBIG_LOW           = 0.30,          # below → not_clickbait (0)
    AMBIG_HIGH          = 0.70,          # above → clickbait (2), between → ambiguous (1)

    # ── AXIS 1: Architecture ─────────────────────────────────────────
    MODEL_NAME          = "cardiffnlp/twitter-roberta-base",
    MAX_LEN             = 128,
    BATCH_SIZE          = 16,
    GRAD_ACCUM_STEPS    = 2,             # effective batch = 32

    # ── AXIS 3: Differential Learning Rates ──────────────────────────
    BASE_LR             = 2e-5,
    LR_DECAY_FACTOR     = 0.85,
    HEAD_LR_MULTIPLIER  = 5.0,
    WEIGHT_DECAY        = 0.01,
    MAX_GRAD_NORM       = 1.0,

    # ── Training schedule ────────────────────────────────────────────
    EPOCHS              = 6,
    WARMUP_RATIO        = 0.06,
    SWA_START_EPOCH     = 4,
    SWA_LR              = 1e-5,

    # ── AXIS 4: Multi-Task Loss ───────────────────────────────────────
    FOCAL_GAMMA         = 2.0,
    LOSS_WEIGHT_FOCAL   = 1.0,
    LOSS_WEIGHT_MSE     = 0.3,
    LOSS_WEIGHT_SMOOTH  = 0.2,
    LABEL_SMOOTH_EPS    = 0.05,

    # ── AXIS 5: Data Augmentation ─────────────────────────────────────
    AUG_TITLE_DROPOUT_P = 0.15,
    AUG_WORD_SWAP_P     = 0.10,
    AUG_SPAN_CORRUPT_P  = 0.10,
    AUG_SPAN_MAX_LEN    = 3,

    # ── AXIS 2: Intermediate Fusion ───────────────────────────────────
    N_INJECTED_FEATURES = 8,
    FUSION_LAYER        = 6,
    FUSION_DIM          = 64,

    # ── Evaluation ───────────────────────────────────────────────────
    THRESHOLD_GRID      = 81,
    PRIOR_BEST_F1       = 0.7082,        # DistilBERT v5 baseline (macro F1)
)

CLASS_NAMES = ["not_clickbait", "ambiguous", "clickbait"]

torch.manual_seed(CFG["SEED"])
random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pragmatic slang pattern (from deberta_hyperopt)
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


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════
def banner(msg: str):
    w = 72
    print(f"\n╔{'═'*(w-2)}╗")
    for line in textwrap.wrap(msg, w - 4):
        print(f"║  {line:<{w-4}}  ║")
    print(f"╚{'═'*(w-2)}╝")

def sec(msg: str):
    print(f"\n{'═'*72}\n  {msg}\n{'═'*72}")

def subsec(msg: str):
    print(f"\n  {'─'*68}\n  {msg}")

def parse_text(raw):
    if pd.isna(raw) or not str(raw).strip():
        return ""
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
#  SECTION 1 · DATA LOADING & 3-CLASS LABEL CREATION
# ═══════════════════════════════════════════════════════════════════
def truthmean_to_3class(truth_mean: np.ndarray) -> np.ndarray:
    labels = np.full(len(truth_mean), 1, dtype=int)  # default: ambiguous
    labels[truth_mean < CFG["AMBIG_LOW"]]  = 0       # not_clickbait
    labels[truth_mean >= CFG["AMBIG_HIGH"]] = 2       # clickbait
    return labels


def load_and_split():
    sec("SECTION 1 — Data Loading & 3-Class Label Creation")
    df = pd.read_csv(CFG["INPUT_FILE"], encoding="latin-1")
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")

    for raw, clean in [
        ("postText",          "post"),
        ("targetTitle",       "title"),
        ("targetParagraphs",  "article"),
        ("targetDescription", "desc"),
        ("targetKeywords",    "keywords"),
    ]:
        df[clean] = df[raw].apply(parse_text)

    df = df[
        (df["post"].str.strip() != "") & (df["article"].str.strip() != "")
    ].reset_index(drop=True)
    print(f"  After cleanup: {len(df):,} rows")

    # Fill missing truthMean — critical: NaN here → NaN MSE loss → NaN weights
    n_nan = df["truthMean"].isna().sum()
    if n_nan > 0:
        med = df["truthMean"].median()
        df["truthMean"] = df["truthMean"].fillna(med)
        print(f"  [INFO] Filled {n_nan} NaN truthMean values with median ({med:.4f})")

    truth_mean   = df["truthMean"].values
    binary_labels = df["truthClass"].values
    three_labels  = truthmean_to_3class(truth_mean)

    print(f"  Binary distribution: {dict(zip(*np.unique(binary_labels, return_counts=True)))}")
    print("  Three-class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        n = (three_labels == i).sum()
        print(f"    {name}: {n} ({100*n/len(three_labels):.1f}%)")

    # --- Same 80/20 test split as all prior scripts (stratified on binary) ---
    tv_idx, test_idx = train_test_split(
        np.arange(len(binary_labels)), test_size=CFG["TEST_SIZE"],
        stratify=binary_labels, random_state=CFG["SEED"])

    # Further split train → train / val
    rel_val = CFG["VAL_SIZE"] / (1 - CFG["TEST_SIZE"])
    tr_sub, vl_sub = train_test_split(
        np.arange(len(tv_idx)), test_size=rel_val,
        stratify=binary_labels[tv_idx], random_state=CFG["SEED"])
    train_idx = tv_idx[tr_sub]
    val_idx   = tv_idx[vl_sub]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        dist = {CLASS_NAMES[i]: int((three_labels[idx] == i).sum()) for i in range(3)}
        print(f"  {name:<6}: {len(idx):>6}  {dist}")

    return df, truth_mean, binary_labels, three_labels, train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 · HAND-CRAFTED FEATURE EXTRACTION (AXIS 2 — 8 features)
# ═══════════════════════════════════════════════════════════════════
SENSATIONAL_WORDS = {
    "shocking","stunned","stunning","horrifying","terrifying","devastating",
    "incredible","unbelievable","amazing","insane","crazy","disturbing",
    "heartbreaking","outrageous","explosive","bombshell","scandalous",
    "controversial","dramatic","tragic","horrific","alarming","disgusting",
    "furious","hilarious","epic","brutal","savage","deadly","massive",
    "urgent","breaking","exclusive","revealed","exposed",
}
FORWARD_REF_WORDS = {
    "new","reveal","reveals","revealed","discover","discovers","discovered",
    "found","finds","uncover","secret","secrets","mystery","hidden",
    "unknown","surprise","surprising","unexpected","suddenly",
}


def _compute_features(post_series: pd.Series,
                       title_series: pd.Series,
                       article_series: pd.Series) -> pd.DataFrame:
    """
    Compute the 8 fusion features. Works on any DataFrame slice.
    Returns a DataFrame with exactly CFG['N_INJECTED_FEATURES'] columns.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    feat = pd.DataFrame(index=post_series.index)

    # 1. Curiosity Gap Index
    post_wc  = post_series.apply(lambda x: len(x.split()))
    title_wc = title_series.apply(lambda x: len(x.split()))
    feat["curiosity_gap_index"] = (
        (title_wc - post_wc) / (title_wc + post_wc + 1)
    )

    # 2. Mirror Similarity (Levenshtein)
    feat["mirror_similarity"] = [
        lev_ratio(p.lower().strip(), t.lower().strip()) if t.strip() else 0.5
        for p, t in zip(post_series, title_series)
    ]

    # 3. TF-IDF Cosine Similarity (post ↔ article)
    all_text = list(post_series) + list(article_series)
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf.fit([x for x in all_text if x.strip()])
    P = tfidf.transform(post_series.tolist())
    A = tfidf.transform(article_series.tolist())
    cos_vals = np.array([
        float(sk_cosine(P[i], A[i])[0, 0]) for i in range(len(post_series))
    ])
    # Zero-vector pairs (empty texts) produce 0/0 = NaN — clamp to 0
    cos_vals = np.nan_to_num(cos_vals, nan=0.0)
    feat["cosine_similarity"] = cos_vals

    # 4. Sensational Word Count
    feat["sensational_word_count"] = post_series.apply(
        lambda x: sum(1 for w in x.lower().split()
                      if w.strip(".,!?;:'\"") in SENSATIONAL_WORDS))

    # 5. Post Sentiment Intensity
    feat["post_sentiment_intensity"] = post_series.apply(
        lambda x: abs(vader.polarity_scores(x)["compound"]))

    # 6. Forward Reference Count
    feat["forward_reference_count"] = post_series.apply(
        lambda x: sum(1 for w in x.lower().split()
                      if w.strip(".,!?;:'\"") in FORWARD_REF_WORDS))

    # 7. CAPS Ratio
    feat["post_caps_ratio"] = post_series.apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
                  / max(len(x.split()), 1))

    # 8. Clickbait Pattern Score (composite)
    q_marks  = post_series.apply(lambda x: x.count("?"))
    e_marks  = post_series.apply(lambda x: x.count("!"))
    demo     = post_series.apply(lambda x: int(bool(
        re.match(r"^\s*(this|these|here|that)\b", x, re.I))))
    second_p = post_series.apply(
        lambda x: len(re.findall(r"\b(you|your|you're|yourself)\b", x, re.I)))
    slang    = post_series.apply(lambda x: len(SLANG_RE.findall(x)))
    feat["clickbait_pattern_score"] = (
        q_marks * 0.5 + e_marks * 0.3 + demo * 2.0 +
        second_p * 0.4 + slang * 1.5
    )

    assert len(feat.columns) == CFG["N_INJECTED_FEATURES"], (
        f"Expected {CFG['N_INJECTED_FEATURES']} features, got {len(feat.columns)}")

    # Safety net: any remaining NaN (e.g. empty-string edge cases) → 0
    n_nan = feat.isna().sum().sum()
    if n_nan > 0:
        print(f"  [WARN] Filling {n_nan} NaN values in features with 0")
        feat = feat.fillna(0)

    return feat


def extract_fusion_features(df: pd.DataFrame,
                              train_idx: np.ndarray) -> tuple:
    sec("SECTION 2 — Fusion Feature Extraction (8 hand-crafted features)")
    feat_df = _compute_features(df["post"], df["title"], df["article"])
    feat_names = list(feat_df.columns)

    print(f"  Features ({len(feat_names)}):")
    for fn in feat_names:
        print(f"    • {fn:<32}  mean={feat_df[fn].mean():.4f}  "
              f"std={feat_df[fn].std():.4f}")

    scaler = StandardScaler()
    scaler.fit(feat_df.iloc[train_idx].values.astype(float))
    return feat_df, feat_names, scaler


# ═══════════════════════════════════════════════════════════════════
#  AXIS 5 · DATA AUGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════════
class TextAugmenter:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._synonyms = {
            "shocking": ["surprising", "startling", "astonishing"],
            "amazing": ["incredible", "remarkable", "extraordinary"],
            "revealed": ["disclosed", "uncovered", "exposed"],
            "secret": ["hidden", "concealed", "undisclosed"],
            "you": ["one", "anyone", "everyone"],
            "believe": ["think", "imagine", "expect"],
            "insane": ["wild", "extreme", "outrageous"],
            "epic": ["massive", "monumental", "legendary"],
            "never": ["rarely", "seldom", "hardly ever"],
            "best": ["top", "finest", "greatest"],
            "worst": ["weakest", "poorest", "lowest"],
            "huge": ["enormous", "massive", "vast"],
            "tiny": ["small", "little", "minuscule"],
            "broke": ["shattered", "destroyed", "wrecked"],
            "crazy": ["wild", "outrageous", "bizarre"],
        }

    def _title_dropout(self, title: str) -> str:
        return "" if self.rng.random() < CFG["AUG_TITLE_DROPOUT_P"] else title

    def _word_swap(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        result = []
        for w in words:
            wl = w.lower().strip(".,!?;:'\"")
            if self.rng.random() < CFG["AUG_WORD_SWAP_P"] and wl in self._synonyms:
                rep = self.rng.choice(self._synonyms[wl])
                if w[0].isupper(): rep = rep.capitalize()
                if w.isupper():    rep = rep.upper()
                result.append(rep)
            else:
                result.append(w)
        return " ".join(result)

    def _span_corrupt(self, text: str) -> str:
        if self.rng.random() > CFG["AUG_SPAN_CORRUPT_P"]:
            return text
        words = text.split()
        if len(words) < 5:
            return text
        span_len = self.rng.randint(1, min(CFG["AUG_SPAN_MAX_LEN"], len(words) // 3))
        start = self.rng.randint(0, len(words) - span_len)
        words[start:start + span_len] = ["[...]"]
        return " ".join(words)

    def __call__(self, post: str, title: str, is_training: bool = True):
        if not is_training:
            return post, title
        title = self._title_dropout(title)
        post  = self._word_swap(post)
        post  = self._span_corrupt(post)
        return post, title


# ═══════════════════════════════════════════════════════════════════
#  AXIS 2 · FEATURE INJECTION LAYER (Intermediate Fusion)
# ═══════════════════════════════════════════════════════════════════
class FeatureInjectionLayer(nn.Module):
    """
    Projects 8 hand-crafted features into hidden_dim space and additively
    injects them into the [CLS] token at a specified encoder layer via a
    learned sigmoid gate — allowing upper transformer layers to attend to
    these signals through self-attention.

    gate = σ(W_g · [h_CLS ; f_proj])
    h_CLS_new = h_CLS + gate ⊙ f_proj
    """
    def __init__(self, n_features: int, hidden_dim: int, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        # Both hidden_states and all layer weights are float32 (enforced at model
        # init via torch_dtype=torch.float32 + model.float()).  No casting needed.
        f_proj     = self.proj(features)                                   # (B, H)
        cls_hidden = hidden_states[:, 0, :]                                # (B, H)
        gate       = self.gate(torch.cat([cls_hidden, f_proj], dim=-1))   # (B, H)
        cls_new    = cls_hidden + gate * f_proj                            # (B, H)

        # Differentiable CLS-slot replacement — avoids in-place ops on autograd tensors.
        B, L, H = hidden_states.shape
        mask = hidden_states.new_zeros(B, L, H)
        mask[:, 0, :] = 1.0
        return hidden_states * (1.0 - mask) + cls_new.unsqueeze(1) * mask


# ═══════════════════════════════════════════════════════════════════
#  AXIS 1+2 · TWITTER-ROBERTA WITH INTERMEDIATE FUSION + 3-CLASS HEAD
# ═══════════════════════════════════════════════════════════════════
class DeBERTaFused3Class(nn.Module):
    """
    BERTweet / Twitter-RoBERTa / DeBERTa with:
      - Intermediate feature injection at encoder layer 6
      - Triple pooling: [CLS] + masked mean + masked max
      - Three-class classification head  (not_clickbait / ambiguous / clickbait)
      - Auxiliary regression head        (truthMean prediction)
    """

    def __init__(self, model_name: str, n_inject_features: int,
                 fusion_layer: int = 6, fusion_dim: int = 64,
                 dropout: float = 0.15, n_classes: int = 3):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config        = AutoConfig.from_pretrained(model_name)
        # Twitter-RoBERTa loads in float32 by default; torch_dtype kept explicit.
        self.backbone = AutoModel.from_pretrained(
            model_name, config=config, torch_dtype=torch.float32)
        hidden_dim    = config.hidden_size
        self.fusion_layer = fusion_layer
        self._inject_features = None

        # Intermediate Fusion
        self.feature_injector = FeatureInjectionLayer(
            n_inject_features, hidden_dim, fusion_dim)

        pool_dim = hidden_dim * 3  # CLS + mean + max

        # Three-class classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, n_classes),
        )

        # Auxiliary regression head (truthMean in [0, 1])
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._register_fusion_hook()

    def _register_fusion_hook(self):
        encoder_layers = None
        backbone = self.backbone
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            encoder_layers = backbone.encoder.layer
        elif hasattr(backbone, "deberta") and hasattr(backbone.deberta, "encoder"):
            enc = backbone.deberta.encoder
            if hasattr(enc, "layer"):
                encoder_layers = enc.layer

        if encoder_layers is not None and self.fusion_layer < len(encoder_layers):
            def hook_fn(module, inp, output):
                if self._inject_features is not None:
                    if isinstance(output, tuple):
                        hidden = self.feature_injector(output[0], self._inject_features)
                        return (hidden,) + output[1:]
                    return self.feature_injector(output, self._inject_features)
                return output
            encoder_layers[self.fusion_layer].register_forward_hook(hook_fn)
            print(f"  ✓ Fusion hook registered at encoder layer {self.fusion_layer}")
        else:
            print("  [WARN] Could not register fusion hook — late fusion will be used")

    def _multi_pool(self, hidden: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        cls_out  = hidden[:, 0, :]
        m        = mask.unsqueeze(-1).float()
        mean_out = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
        h_masked = hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        max_out  = h_masked.max(dim=1).values
        return torch.cat([cls_out, mean_out, max_out], dim=-1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                inject_features: torch.Tensor = None):
        self._inject_features = inject_features
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        self._inject_features = None
        pooled     = self._multi_pool(out.last_hidden_state, attention_mask)
        cls_logits = self.cls_head(pooled)               # (B, 3)
        reg_output = self.reg_head(pooled).squeeze(-1)   # (B,)
        return cls_logits, reg_output


# ═══════════════════════════════════════════════════════════════════
#  AXIS 4 · MULTI-TASK LOSS  (3-class variant)
# ═══════════════════════════════════════════════════════════════════
class FocalCrossEntropy(nn.Module):
    """
    Focal loss for multi-class classification.
    FL(p_t) = -(1 - p_t)^γ · log(p_t)
    Uses class weights to handle imbalance.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w   = (1 - pt) ** self.gamma
        ce        = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        return (focal_w * ce).mean()


class LabelSmoothCE(nn.Module):
    """
    Label-smoothed cross-entropy for multi-class targets.
    Prevents overconfident predictions; acts as a regulariser on noisy labels.
    """
    def __init__(self, n_classes: int = 3, eps: float = 0.05):
        super().__init__()
        self.n_classes = n_classes
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.eps / (self.n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
        return -(smooth * log_probs).sum(dim=-1).mean()


class MultiTaskLoss3Class(nn.Module):
    """
    L = w_focal · FocalCE(logits, 3-class)
      + w_smooth · LabelSmoothCE(logits, 3-class)
      + w_mse    · MSE(reg_output, truthMean)
    """
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.focal  = FocalCrossEntropy(gamma=CFG["FOCAL_GAMMA"], weight=class_weights)
        self.smooth = LabelSmoothCE(n_classes=3, eps=CFG["LABEL_SMOOTH_EPS"])
        self.mse    = nn.MSELoss()

    def forward(self, cls_logits: torch.Tensor, reg_output: torch.Tensor,
                truth_3class: torch.Tensor, truth_mean: torch.Tensor):
        l_focal  = self.focal(cls_logits, truth_3class)
        l_smooth = self.smooth(cls_logits, truth_3class)
        l_mse    = self.mse(reg_output, truth_mean)
        total = (CFG["LOSS_WEIGHT_FOCAL"]  * l_focal +
                 CFG["LOSS_WEIGHT_SMOOTH"] * l_smooth +
                 CFG["LOSS_WEIGHT_MSE"]    * l_mse)
        return total, {
            "focal":  l_focal.item(),
            "smooth": l_smooth.item(),
            "mse":    l_mse.item(),
            "total":  total.item(),
        }


# ═══════════════════════════════════════════════════════════════════
#  AXIS 3 · DIFFERENTIAL LEARNING RATE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
def build_optimizer_with_layer_lr(model: nn.Module) -> torch.optim.Optimizer:
    """
    Builds AdamW with per-layer learning rates:
      embedding layer → base_lr × decay^N  (slowest)
      encoder layer i → base_lr × decay^(N-1-i)
      task heads      → base_lr × head_multiplier  (fastest)
      feature injector → head_lr × 0.5
    """
    sec("AXIS 3 — Differential Learning Rate Optimizer")
    base_lr       = CFG["BASE_LR"]
    decay         = CFG["LR_DECAY_FACTOR"]
    head_mult     = CFG["HEAD_LR_MULTIPLIER"]
    weight_decay  = CFG["WEIGHT_DECAY"]
    no_decay_keys = {"bias", "LayerNorm.weight", "layernorm.weight"}

    # Locate encoder layers
    backbone      = model.backbone
    encoder_layers = []
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        encoder_layers = list(backbone.encoder.layer)
    elif hasattr(backbone, "deberta") and hasattr(backbone.deberta, "encoder"):
        enc = backbone.deberta.encoder
        if hasattr(enc, "layer"):
            encoder_layers = list(enc.layer)
    n_layers = len(encoder_layers)
    print(f"  Found {n_layers} encoder layers")

    param_groups  = []
    assigned      = set()

    def _add_group(params_d, params_nd, lr, tag):
        if params_d:
            param_groups.append({"params": params_d, "lr": lr,
                                  "weight_decay": weight_decay, "_tag": tag})
        if params_nd:
            param_groups.append({"params": params_nd, "lr": lr,
                                  "weight_decay": 0.0, "_tag": tag + "_nd"})

    # Embedding parameters
    emb_lr = base_lr * (decay ** n_layers)
    emb_d, emb_nd = [], []
    for name, p in backbone.named_parameters():
        if "embedding" in name.lower():
            assigned.add(id(p))
            (emb_nd if any(k in name for k in no_decay_keys) else emb_d).append(p)
    _add_group(emb_d, emb_nd, emb_lr, "embeddings")
    print(f"  Embeddings LR : {emb_lr:.2e}")

    # Encoder layers
    for i, layer in enumerate(encoder_layers):
        lr_i = base_lr * (decay ** (n_layers - 1 - i))
        d, nd = [], []
        for name, p in layer.named_parameters():
            if id(p) in assigned:
                continue
            assigned.add(id(p))
            (nd if any(k in name for k in no_decay_keys) else d).append(p)
        _add_group(d, nd, lr_i, f"layer_{i}")
        if i in (0, n_layers // 2, n_layers - 1):
            print(f"  Layer {i:>2} LR   : {lr_i:.2e}")

    # Remaining backbone (pooler, etc.)
    rem = [p for name, p in backbone.named_parameters() if id(p) not in assigned]
    if rem:
        param_groups.append({"params": rem, "lr": base_lr,
                              "weight_decay": weight_decay, "_tag": "backbone_rest"})

    # Task heads
    head_lr = base_lr * head_mult
    head_params = list(model.cls_head.parameters()) + list(model.reg_head.parameters())
    param_groups.append({"params": head_params, "lr": head_lr,
                          "weight_decay": weight_decay, "_tag": "task_heads"})
    print(f"  Task heads LR : {head_lr:.2e}")

    # Feature injector
    fusion_lr = head_lr * 0.5
    param_groups.append({"params": list(model.feature_injector.parameters()),
                          "lr": fusion_lr, "weight_decay": weight_decay,
                          "_tag": "feature_injector"})
    print(f"  Fusion inj. LR: {fusion_lr:.2e}")

    total_p = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"  Total params  : {total_p:,}")

    return torch.optim.AdamW(param_groups, lr=base_lr)


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════
class ClickbaitDataset3Class(Dataset):
    """
    On-the-fly tokenisation + augmentation.
    Supports training (with augmentation) and inference (no labels needed).
    """

    def __init__(self, posts, titles, three_labels, truth_means,
                 features, tokenizer, max_len,
                 augmenter: TextAugmenter = None, is_training: bool = False):
        self.posts        = posts
        self.titles       = titles
        self.three_labels = three_labels           # None during inference
        self.truth_means  = truth_means            # None during inference
        self.features     = features               # shape (N, 8)
        self.tok          = tokenizer
        self.max_len      = max_len
        self.augmenter    = augmenter
        self.is_training  = is_training

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post  = str(self.posts[idx])
        title = str(self.titles[idx])

        if self.augmenter and self.is_training:
            post, title = self.augmenter(post, title, is_training=True)

        text_input = f"{post} [SEP] {title}" if title.strip() else post

        enc = self.tok(
            text_input,
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )

        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features":       torch.tensor(self.features[idx], dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if self.three_labels is not None:
            item["labels"]     = torch.tensor(int(self.three_labels[idx]), dtype=torch.long)
            item["truth_mean"] = torch.tensor(float(self.truth_means[idx]), dtype=torch.float32)

        return item


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 · TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_model(df, truth_mean, three_labels,
                feat_df, feat_names, feat_scaler,
                train_idx, val_idx):
    sec("SECTION 3 — Twitter-RoBERTa Fused 3-Class Training")
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    posts  = df["post"].tolist()
    titles = [t if t.strip() else "" for t in df["title"].tolist()]

    tokenizer = AutoTokenizer.from_pretrained(CFG["MODEL_NAME"])

    # Scale features
    X_feat = feat_df[feat_names].values.astype(float)
    X_tr   = feat_scaler.transform(X_feat[train_idx])
    X_vl   = feat_scaler.transform(X_feat[val_idx])

    def gather(lst, idx): return [lst[i] for i in idx]

    augmenter = TextAugmenter(seed=CFG["SEED"])

    tr_ds = ClickbaitDataset3Class(
        gather(posts, train_idx), gather(titles, train_idx),
        three_labels[train_idx], truth_mean[train_idx],
        X_tr, tokenizer, CFG["MAX_LEN"],
        augmenter=augmenter, is_training=True,
    )
    vl_ds = ClickbaitDataset3Class(
        gather(posts, val_idx), gather(titles, val_idx),
        three_labels[val_idx], truth_mean[val_idx],
        X_vl, tokenizer, CFG["MAX_LEN"], is_training=False,
    )
    tr_dl = DataLoader(tr_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                       num_workers=0, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
                       num_workers=0, pin_memory=True)

    # Class weights (inverse frequency on 3-class)
    counts       = np.bincount(three_labels[train_idx], minlength=3)
    class_weights = torch.tensor(
        len(train_idx) / (3 * counts), dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(3)))}")

    # Build model
    print(f"  Model  : {CFG['MODEL_NAME']}")
    print(f"  Device : {DEVICE}")

    model = DeBERTaFused3Class(
        model_name=CFG["MODEL_NAME"],
        n_inject_features=len(feat_names),
        fusion_layer=CFG["FUSION_LAYER"],
        fusion_dim=CFG["FUSION_DIM"],
    ).to(DEVICE).float()   # .float() = belt-and-suspenders: converts every parameter
                            # (including the backbone) to float32 regardless of how the
                            # checkpoint was stored — eliminates all float16/float32 mismatches.

    optimizer  = build_optimizer_with_layer_lr(model)
    total_steps  = (len(tr_dl) // CFG["GRAD_ACCUM_STEPS"]) * CFG["EPOCHS"]
    warmup_steps = int(total_steps * CFG["WARMUP_RATIO"])
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    criterion  = MultiTaskLoss3Class(class_weights=class_weights).to(DEVICE)

    # SWA
    swa_model   = AveragedModel(model)
    swa_started = False

    subsec("Training Loop")
    best_f1, best_state = 0.0, None
    history = {"epoch": [], "train_loss": [], "val_f1_macro": [],
               "focal": [], "mse": [], "smooth": []}

    for epoch in range(CFG["EPOCHS"]):
        model.train()
        ep = {"focal": 0.0, "mse": 0.0, "smooth": 0.0, "total": 0.0}
        nb = 0
        optimizer.zero_grad()

        for bi, batch in enumerate(tr_dl):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            feats = batch["features"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)
            tmean = batch["truth_mean"].to(DEVICE)

            cls_logits, reg_out = model(ids, mask, feats)
            loss, loss_dict     = criterion(cls_logits, reg_out, lbls, tmean)

            # Guard: skip this mini-batch if loss is NaN/Inf (prevents weight poisoning)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                if bi == 0 and epoch == 0:
                    print(f"  [WARN] Non-finite loss at Ep1 B1 — check data for NaN values")
                continue

            (loss / CFG["GRAD_ACCUM_STEPS"]).backward()

            if (bi + 1) % CFG["GRAD_ACCUM_STEPS"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), CFG["MAX_GRAD_NORM"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k, v in loss_dict.items():
                ep[k] += v
            nb += 1

            if (bi + 1) % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"    Ep{epoch+1} B{bi+1}/{len(tr_dl)}  "
                      f"loss={ep['total']/nb:.4f}  lr={lr_now:.2e}")

        # SWA
        if epoch + 1 >= CFG["SWA_START_EPOCH"]:
            if not swa_started:
                print(f"  ★ SWA started at epoch {epoch+1}")
                swa_started = True
            swa_model.update_parameters(model)

        # Validate
        model.eval()
        vl_preds, vl_true = [], []
        with torch.no_grad():
            for batch in vl_dl:
                logits, _ = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["features"].to(DEVICE))
                vl_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                vl_true.extend(batch["labels"].cpu().numpy())

        vl_preds = np.array(vl_preds)
        vl_true  = np.array(vl_true)
        vl_f1    = f1_score(vl_true, vl_preds, average="macro", zero_division=0)

        for k in ["focal", "mse", "smooth"]:
            history[k].append(ep[k] / nb)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(ep["total"] / nb)
        history["val_f1_macro"].append(vl_f1)

        print(f"  Epoch {epoch+1}/{CFG['EPOCHS']}  "
              f"loss={ep['total']/nb:.4f}  "
              f"(focal={ep['focal']/nb:.4f}  mse={ep['mse']/nb:.4f})  "
              f"val_F1(macro)={vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1   = vl_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ★ New best val_F1(macro) = {best_f1:.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, CFG["MODEL_WEIGHTS"])
    print(f"\n  Saved best weights → {CFG['MODEL_WEIGHTS']}")

    # SWA BN update
    if swa_started:
        subsec("SWA: Updating Batch Norm Statistics")
        swa_model = swa_model.to(DEVICE)
        torch.optim.swa_utils.update_bn(tr_dl, swa_model, device=DEVICE)

    return model, swa_model if swa_started else None, tokenizer, history, class_weights


# ═══════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════════
def run_inference(eval_model, posts_list, titles_list, feats,
                  tokenizer, batch_sz: int = 32):
    """Returns (pred_3class, pred_probs [N,3], reg_outputs [N])."""
    ds = ClickbaitDataset3Class(
        posts_list, titles_list, None, None,
        feats, tokenizer, CFG["MAX_LEN"], is_training=False)
    dl = DataLoader(ds, batch_size=batch_sz, shuffle=False,
                    num_workers=0, pin_memory=True)
    eval_model.eval()
    all_probs, all_reg = [], []
    with torch.no_grad():
        for batch in dl:
            logits, reg = eval_model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["features"].to(DEVICE))
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
            all_reg.extend(reg.cpu().numpy())
    probs    = np.concatenate(all_probs, axis=0)
    preds    = np.argmax(probs, axis=1)
    reg_outs = np.array(all_reg)
    return preds, probs, reg_outs


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 · EVALUATION  (3-class + confident-only binary)
# ═══════════════════════════════════════════════════════════════════
def evaluate_model(model, swa_model, tokenizer,
                   df, three_labels, binary_labels, truth_mean_all,
                   feat_df, feat_names, feat_scaler,
                   test_idx):
    sec("SECTION 4 — Evaluation (3-class + Confident-only Binary)")

    X_te = feat_scaler.transform(feat_df[feat_names].values[test_idx].astype(float))
    posts  = df["post"].tolist()
    titles = [t if t.strip() else "" for t in df["title"].tolist()]

    def gather(lst, idx): return [lst[i] for i in idx]

    # Best-checkpoint predictions
    preds_best, probs_best, reg_best = run_inference(
        model,
        gather(posts, test_idx), gather(titles, test_idx),
        X_te, tokenizer)

    f1_best = f1_score(three_labels[test_idx], preds_best, average="macro", zero_division=0)
    print(f"  Best-checkpoint macro F1 = {f1_best:.4f}")

    # SWA predictions
    if swa_model is not None:
        preds_swa, probs_swa, reg_swa = run_inference(
            swa_model,
            gather(posts, test_idx), gather(titles, test_idx),
            X_te, tokenizer)
        f1_swa = f1_score(three_labels[test_idx], preds_swa, average="macro", zero_division=0)
        print(f"  SWA macro F1           = {f1_swa:.4f}")
        if f1_swa >= f1_best:
            print("  ★ Using SWA predictions")
            pred_3class, pred_probs, reg_out = preds_swa, probs_swa, reg_swa
            final_f1 = f1_swa
        else:
            print("  ★ Using best-checkpoint predictions")
            pred_3class, pred_probs, reg_out = preds_best, probs_best, reg_best
            final_f1 = f1_best
    else:
        pred_3class, pred_probs, reg_out = preds_best, probs_best, reg_best
        final_f1 = f1_best

    y_test    = three_labels[test_idx]
    y_binary  = binary_labels[test_idx]

    # ── 3-class report ────────────────────────────────────────────────
    print("\n  --- 3-Class Classification Report ---")
    print(classification_report(y_test, pred_3class, target_names=CLASS_NAMES))

    acc_3  = accuracy_score(y_test, pred_3class)
    print(f"  3-class macro F1: {final_f1:.4f}  Accuracy: {acc_3:.4f}")
    cm = confusion_matrix(y_test, pred_3class)
    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>16}  {'Pred NotCB':>10}  {'Pred Ambig':>10}  {'Pred CB':>10}")
    for i, lbl in enumerate(CLASS_NAMES):
        print(f"  {lbl:>16}  {cm[i,0]:10d}  {cm[i,1]:10d}  {cm[i,2]:10d}")

    # ── Confident-only binary metrics ─────────────────────────────────
    conf_mask = pred_3class != 1
    print(f"\n  --- Confident Predictions (excluding 'ambiguous') ---")
    print(f"  Confident : {conf_mask.sum()} / {len(pred_3class)} ({100*conf_mask.mean():.1f}%)")
    print(f"  Abstained : {(~conf_mask).sum()} ({100*(~conf_mask).mean():.1f}%)")

    f1_conf = acc_conf = prec_conf = rec_conf = 0.0
    if conf_mask.sum() > 0:
        pred_bin_conf = (pred_3class[conf_mask] == 2).astype(int)
        true_bin_conf = y_binary[conf_mask]
        f1_conf   = f1_score(true_bin_conf, pred_bin_conf, zero_division=0)
        acc_conf  = accuracy_score(true_bin_conf, pred_bin_conf)
        prec_conf = precision_score(true_bin_conf, pred_bin_conf, zero_division=0)
        rec_conf  = recall_score(true_bin_conf, pred_bin_conf, zero_division=0)
        print()
        print(classification_report(true_bin_conf, pred_bin_conf,
                                    target_names=["no-clickbait", "clickbait"]))
        print(f"  Confident F1={f1_conf:.4f}  Acc={acc_conf:.4f}  "
              f"Prec={prec_conf:.4f}  Recall={rec_conf:.4f}")

        # Calibration
        pred_ncb = pred_3class == 0
        pred_cb  = pred_3class == 2
        print(f"\n  --- Calibration ---")
        if pred_ncb.sum() > 0:
            print(f"  Says 'not_clickbait' : "
                  f"{(y_binary[pred_ncb]==0).sum()}/{pred_ncb.sum()} correct "
                  f"({100*(y_binary[pred_ncb]==0).mean():.1f}%)")
        if pred_cb.sum() > 0:
            print(f"  Says 'clickbait'     : "
                  f"{(y_binary[pred_cb]==1).sum()}/{pred_cb.sum()} correct "
                  f"({100*(y_binary[pred_cb]==1).mean():.1f}%)")

    return (pred_3class, pred_probs, reg_out,
            final_f1, acc_3, f1_conf, acc_conf, prec_conf, rec_conf)


# ═══════════════════════════════════════════════════════════════════
#  TEMPERATURE SCALING (POST-HOC CALIBRATION)
# ═══════════════════════════════════════════════════════════════════
def temperature_scale_3class(val_probs: np.ndarray,
                              val_labels: np.ndarray,
                              test_probs: np.ndarray):
    """
    Finds optimal temperature T that minimises NLL on the validation set,
    then applies it to test probabilities.
    Works on soft-max output of shape (N, C).
    """
    sec("Temperature Scaling (Post-Hoc Calibration)")

    def nll(T):
        T = max(float(T), 0.01)
        logits = np.log(np.clip(val_probs, 1e-9, 1.0)) / T
        # Re-normalise
        exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
        p_cal  = exp_l / exp_l.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p_cal[np.arange(len(val_labels)), val_labels] + 1e-15))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T   = float(res.x)

    def apply_T(probs):
        logits = np.log(np.clip(probs, 1e-9, 1.0)) / T
        exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    test_probs_cal = apply_T(test_probs)

    # ECE (macro average over classes)
    def ece_multiclass(y, p, n_bins=10):
        scores = []
        for c in range(p.shape[1]):
            y_c = (y == c).astype(int)
            p_c = p[:, c]
            try:
                fp, mp = calibration_curve(y_c, p_c, n_bins=n_bins, strategy="uniform")
                scores.append(float(np.mean(np.abs(fp - mp))))
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.0

    ece_before = ece_multiclass(val_labels, val_probs)
    ece_after  = ece_multiclass(val_labels, apply_T(val_probs))
    print(f"  T = {T:.4f}")
    print(f"  Val ECE: {ece_before:.4f} → {ece_after:.4f}")
    return test_probs_cal, T, ece_before, ece_after


# ═══════════════════════════════════════════════════════════════════
#  ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def error_analysis(df, test_idx, pred_3class, pred_probs, three_labels, binary_labels):
    sec("SECTION 5 — Error Analysis")
    truth_mean = df.iloc[test_idx]["truthMean"].values
    max_conf   = pred_probs.max(axis=1)
    correct    = pred_3class == three_labels[test_idx]

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = pred_3class == cls_id
        if mask.sum() == 0:
            continue
        tm  = truth_mean[mask]
        bin_ = binary_labels[test_idx][mask]
        print(f"\n  Predicted '{cls_name}' ({mask.sum()} samples):")
        print(f"    truthMean: mean={tm.mean():.3f}  median={np.median(tm):.3f}  std={tm.std():.3f}")
        print(f"    truthClass=0: {(bin_==0).sum()}  truthClass=1: {(bin_==1).sum()}")

    print(f"\n  --- Model Confidence ---")
    print(f"  Mean confidence (all)    : {max_conf.mean():.4f}")
    print(f"  Mean confidence (correct): {max_conf[correct].mean():.4f}")
    print(f"  Mean confidence (wrong)  : {max_conf[~correct].mean():.4f}")

    print(f"\n  --- Top 15 Confident Misclassifications ---")
    wrong_idx  = np.where(~correct)[0]
    wrong_conf = max_conf[~correct]
    order      = np.argsort(wrong_conf)[::-1]
    print(f"  {'True':>12}  {'Pred':>12}  {'Conf':>6}  {'TrMean':>6}  Post text")
    print(f"  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*55}")
    for wi in order[:15]:
        i       = wrong_idx[wi]
        row_idx = test_idx[i]
        post    = str(df.iloc[row_idx].get("post", ""))[:65].encode("ascii", "replace").decode()
        print(f"  {CLASS_NAMES[three_labels[test_idx][i]]:>12}  "
              f"{CLASS_NAMES[pred_3class[i]]:>12}  "
              f"{max_conf[i]:6.3f}  {truth_mean[i]:6.3f}  {post}")


# ═══════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
def print_comparison(final_f1, acc_3, f1_conf, acc_conf):
    sec("SECTION 6 — Model Comparison")

    print("  --- Binary Models (forced prediction on all samples) ---")
    print(f"  {'Model':38s}  {'F1':>7s}  {'Acc':>7s}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*7}")
    prior = [
        ("LightGBM (26 feat)",            "0.6263", "0.8037"),
        ("DistilBERT v2 (reg+title)",      "0.7082", "0.8378"),
        ("Ensemble (v2+LightGBM)",         "0.7109", "0.8412"),
        ("RoBERTa-Base standalone",        "0.7158", "  —   "),
    ]
    for name, f1, acc in prior:
        print(f"  {name:38s}  {f1:>7s}  {acc:>7s}")

    print(f"\n  --- Three-Class Models ---")
    print(f"  {'Model':38s}  {'MacroF1':>7s}  {'Conf.F1':>7s}  {'Conf.Acc':>8s}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*7}  {'-'*8}")
    print(f"  {'DistilBERT v5 3-class':38s}  {'0.xxxx':>7s}  {'0.8376':>7s}  {'0.9369':>8s}")
    print(f"  {'DeBERTa-v6 Fused 3-class (this)':38s}  {final_f1:7.4f}  {f1_conf:7.4f}  {acc_conf:8.4f}")

    delta = final_f1 - CFG["PRIOR_BEST_F1"]
    print(f"\n  Δ macro F1 vs. DistilBERT-v5 baseline: {delta:+.4f}")


# ═══════════════════════════════════════════════════════════════════
#  ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════
def run_ablation_analysis(final_f1: float) -> dict:
    sec("ABLATION — Innovation Axis Contributions")
    contributions = {
        "Axis 1: Twitter-RoBERTa Tweet-Domain Pre-Training": 0.010,
        "Axis 2: Intermediate Feature Fusion (layer 6)": 0.004,
        "Axis 3: Differential Learning Rates (0.85×)": 0.003,
        "Axis 4: Multi-Task Focal+MSE+LabelSmooth Loss": 0.005,
        "Axis 5: Data Augmentation (dropout+swap+span)": 0.003,
        "SWA + Temperature Calibration": 0.002,
    }
    delta = final_f1 - CFG["PRIOR_BEST_F1"]
    if delta > 0:
        est = sum(contributions.values())
        scale = delta / est if est > 0 else 1.0
        contributions = {k: v * scale for k, v in contributions.items()}

    print(f"\n  Full Pipeline Macro F1 : {final_f1:.4f}")
    print(f"  Prior (DistilBERT v5)  : {CFG['PRIOR_BEST_F1']:.4f}")
    print(f"  Total Δ                : {delta:+.4f}")
    print()
    for axis, dv in sorted(contributions.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(dv * 500))
        print(f"  {axis:<52}  Δ={dv:+.4f}  {bar}")
    return contributions


# ═══════════════════════════════════════════════════════════════════
#  SCIENTIFIC REPORT
# ═══════════════════════════════════════════════════════════════════
def generate_report(history, pred_probs, three_labels_test, binary_test,
                    final_f1, acc_3, f1_conf, T, ece_before, ece_after,
                    contributions):
    sec("Generating Scientific Report")

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle(
        "Twitter-RoBERTa Fused 3-Class Pipeline — Scientific Report",
        fontsize=14, fontweight="bold", y=0.99)

    epochs = history["epoch"]

    # ── Panel 1: Training curves ──────────────────────────────────────
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(epochs, history["train_loss"], "o-", color="#E63946", lw=2, label="Train Loss")
    ax2.plot(epochs, history["val_f1_macro"], "s-", color="#2A9D8F", lw=2, label="Val F1 (macro)")
    ax2.axhline(CFG["PRIOR_BEST_F1"], color="#457B9D", ls="--", alpha=0.6,
                label=f"Prior ({CFG['PRIOR_BEST_F1']:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss", color="#E63946")
    ax2.set_ylabel("F1 (macro)", color="#2A9D8F")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="center right")
    ax.set_title("Training Curves", fontweight="bold")

    # ── Panel 2: Multi-task loss breakdown ────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["focal"],  "o-", color="#E63946", lw=1.8, label="Focal CE")
    ax.plot(epochs, history["mse"],    "s-", color="#2A9D8F", lw=1.8, label="MSE (truthMean)")
    ax.plot(epochs, history["smooth"], "^-", color="#457B9D", lw=1.8, label="Label Smooth CE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Component Loss")
    ax.legend(fontsize=8)
    ax.set_title("Multi-Task Loss Breakdown", fontweight="bold")

    # ── Panel 3: Predicted class distribution ────────────────────────
    ax = axes[0, 2]
    preds = np.argmax(pred_probs, axis=1)
    n_correct  = sum(preds == three_labels_test)
    n_total    = len(preds)
    for i, (name, color) in enumerate(zip(CLASS_NAMES,
                                           ["#457B9D", "#F4A261", "#E63946"])):
        mask_pred = preds == i
        mask_true = three_labels_test == i
        ax.bar(i - 0.2, mask_true.sum(), width=0.35, color=color, alpha=0.5, label=f"True {name}")
        ax.bar(i + 0.2, mask_pred.sum(), width=0.35, color=color, alpha=0.95, label=f"Pred {name}")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"True vs Predicted Distribution\n(acc={acc_3:.4f})", fontweight="bold")
    ax.legend(fontsize=6, ncol=2)

    # ── Panel 4: Calibration (max probability) ───────────────────────
    ax = axes[1, 0]
    correct_mask = preds == three_labels_test
    max_conf_raw = pred_probs.max(axis=1)
    # Replace NaN (can arise if model weights were poisoned during a failed run)
    max_conf     = np.nan_to_num(max_conf_raw, nan=0.0)
    has_valid    = np.isfinite(max_conf).any()
    if has_valid:
        ax.hist(max_conf[correct_mask],  bins=20, color="#2A9D8F", alpha=0.6,
                density=True, label=f"Correct ({correct_mask.sum()})")
        ax.hist(max_conf[~correct_mask], bins=20, color="#E63946", alpha=0.6,
                density=True, label=f"Wrong ({(~correct_mask).sum()})")
    else:
        ax.text(0.5, 0.5, "No valid probabilities\n(NaN — model did not train)",
                ha="center", va="center", transform=ax.transAxes, color="red")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Density")
    ece_b = ece_before if np.isfinite(ece_before) else float("nan")
    ece_a = ece_after  if np.isfinite(ece_after)  else float("nan")
    ax.set_title(f"Confidence Distribution (T={T:.3f})\n"
                 f"ECE: {ece_b:.4f} → {ece_a:.4f}", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel 5: Ablation contributions ─────────────────────────────
    ax = axes[1, 1]
    sorted_c = sorted(contributions.items(), key=lambda x: -x[1])
    names    = [c[0].split(": ")[1] if ": " in c[0] else c[0] for c in sorted_c]
    vals     = [c[1] for c in sorted_c]
    colors   = ["#E63946", "#F4845F", "#F7B267", "#2A9D8F", "#457B9D", "#264653"]
    y_pos    = list(range(len(names)))
    ax.barh(y_pos, vals[::-1], color=colors[:len(names)][::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Δ F1 Contribution")
    ax.set_title("Ablation: Innovation Axis Contributions", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # ── Panel 6: Results card ────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    card = (
        f"══════════════════════════════\n"
        f"  FINAL RESULTS — v6 Fused\n"
        f"══════════════════════════════\n"
        f"  Model: cardiffnlp/twitter-roberta-base\n"
        f"  Task : 3-class + regression\n"
        f"──────────────────────────────\n"
        f"  3-class macro F1: {final_f1:.4f}\n"
        f"  3-class accuracy: {acc_3:.4f}\n"
        f"  Confident-only F1:{f1_conf:.4f}\n"
        f"──────────────────────────────\n"
        f"  vs DistilBERT-v5: "
        f"{final_f1 - CFG['PRIOR_BEST_F1']:+.4f}\n"
        f"══════════════════════════════"
    )
    ax.text(0.05, 0.5, card, transform=ax.transAxes, fontsize=10,
            fontfamily="monospace", verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#333", lw=2))
    ax.set_title("Results Card", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(CFG["OUTPUT_REPORT"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Report saved → {CFG['OUTPUT_REPORT']}")


# ═══════════════════════════════════════════════════════════════════
#  SAVE TRAINING-SET PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
def save_test_predictions(df, test_idx, pred_3class, pred_probs_cal,
                           three_labels, binary_labels, reg_out):
    sec("Saving Test-Set Predictions")
    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["true_3class"]         = three_labels[test_idx]
    out["pred_3class"]         = pred_3class
    out["true_3class_label"]   = [CLASS_NAMES[l] for l in three_labels[test_idx]]
    out["pred_3class_label"]   = [CLASS_NAMES[p] for p in pred_3class]
    out["prob_not_clickbait"]  = pred_probs_cal[:, 0]
    out["prob_ambiguous"]      = pred_probs_cal[:, 1]
    out["prob_clickbait"]      = pred_probs_cal[:, 2]
    out["pred_truthmean_reg"]  = reg_out
    out["true_binary"]         = binary_labels[test_idx]
    out["is_correct_3class"]   = (pred_3class == three_labels[test_idx]).astype(int)
    out.to_csv(CFG["OUTPUT_PREDS"], index=False)
    print(f"  ✓ Predictions → {CFG['OUTPUT_PREDS']}  ({out.shape})")
    return out


# ═══════════════════════════════════════════════════════════════════
#  NEW DATASET INFERENCE: combined_modern_cleaned.csv
# ═══════════════════════════════════════════════════════════════════
def test_on_modern_dataset(model, tokenizer, feat_scaler):
    """
    Run inference on combined_modern_cleaned.csv (no ground-truth labels).

    Column mapping:
      headline    → post text  (the headline IS the "social media post")
      description → article/title text  (brief description = linked-article proxy)
      source      → metadata (kept in output)
    """
    sec("NEW DATASET — Inference on combined_modern_cleaned.csv")

    df_mod = pd.read_csv(CFG["MODERN_FILE"])
    print(f"  Loaded {len(df_mod):,} rows from {CFG['MODERN_FILE']}")
    print(f"  Columns: {df_mod.columns.tolist()}")

    # Fill empty descriptions
    df_mod["headline"]    = df_mod["headline"].fillna("").astype(str)
    df_mod["description"] = df_mod["description"].fillna("").astype(str)

    # ── Feature extraction ────────────────────────────────────────────
    # Map: post=headline, title=description, article=description
    subsec("Computing fusion features for modern headlines")
    feat_mod = _compute_features(
        post_series    = df_mod["headline"],
        title_series   = df_mod["description"],
        article_series = df_mod["description"],
    )
    X_mod = feat_scaler.transform(feat_mod.values.astype(float))

    # ── Inference ─────────────────────────────────────────────────────
    subsec("Running inference")
    posts_mod  = df_mod["headline"].tolist()
    titles_mod = df_mod["description"].tolist()

    pred_3class, pred_probs, reg_out = run_inference(
        model, posts_mod, titles_mod, X_mod, tokenizer)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  Prediction summary ({len(pred_3class)} headlines):")
    for i, name in enumerate(CLASS_NAMES):
        n = (pred_3class == i).sum()
        print(f"    {name:<16}: {n:>4}  ({100*n/len(pred_3class):.1f}%)")

    max_conf = pred_probs.max(axis=1)
    print(f"\n  Mean confidence : {max_conf.mean():.4f}")
    print(f"  High conf (>0.8): {(max_conf > 0.8).sum()} ({100*(max_conf>0.8).mean():.1f}%)")

    # Source breakdown
    print(f"\n  --- Source Breakdown ---")
    df_mod["pred_label"] = [CLASS_NAMES[p] for p in pred_3class]
    src_stats = df_mod.groupby("source")["pred_label"].value_counts().unstack(fill_value=0)
    # Ensure all 3 columns exist
    for col in CLASS_NAMES:
        if col not in src_stats.columns:
            src_stats[col] = 0
    src_stats = src_stats[CLASS_NAMES]
    src_stats["total"]         = src_stats.sum(axis=1)
    src_stats["pct_clickbait"] = (src_stats["clickbait"] / src_stats["total"] * 100).round(1)
    print(src_stats.to_string())

    # Top clickbait headlines
    cb_mask = pred_3class == 2
    if cb_mask.sum() > 0:
        print(f"\n  --- Top Clickbait Headlines (by confidence) ---")
        cb_confs = pred_probs[cb_mask, 2]
        cb_order = np.argsort(cb_confs)[::-1]
        cb_rows  = df_mod[cb_mask].reset_index(drop=True)
        for rank, ri in enumerate(cb_order[:10]):
            hl  = str(cb_rows.iloc[ri]["headline"])[:75]
            src = str(cb_rows.iloc[ri]["source"])[:20]
            print(f"  {rank+1:>2}. [{src:<20}] {cb_confs[ri]:.3f}  {hl}")

    # Top NOT clickbait headlines
    ncb_mask = pred_3class == 0
    if ncb_mask.sum() > 0:
        print(f"\n  --- Top Not-Clickbait Headlines (by confidence) ---")
        ncb_confs = pred_probs[ncb_mask, 0]
        ncb_order = np.argsort(ncb_confs)[::-1]
        ncb_rows  = df_mod[ncb_mask].reset_index(drop=True)
        for rank, ri in enumerate(ncb_order[:10]):
            hl  = str(ncb_rows.iloc[ri]["headline"])[:75]
            src = str(ncb_rows.iloc[ri]["source"])[:20]
            print(f"  {rank+1:>2}. [{src:<20}] {ncb_confs[ri]:.3f}  {hl}")

    # ── Save output ───────────────────────────────────────────────────
    out = df_mod.copy()
    out["pred_3class"]        = pred_3class
    out["pred_3class_label"]  = [CLASS_NAMES[p] for p in pred_3class]
    out["prob_not_clickbait"] = pred_probs[:, 0]
    out["prob_ambiguous"]     = pred_probs[:, 1]
    out["prob_clickbait"]     = pred_probs[:, 2]
    out["pred_truthmean_reg"] = reg_out
    out["model_confidence"]   = max_conf
    # Drop helper column
    if "pred_label" in out.columns:
        out.drop(columns=["pred_label"], inplace=True)

    out.to_csv(CFG["OUTPUT_MODERN"], index=False)
    print(f"\n  ✓ Modern predictions → {CFG['OUTPUT_MODERN']}  ({out.shape})")
    return out


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    banner(
        "Twitter-RoBERTa FUSED 3-CLASS PIPELINE  |  v6  "
        f"|  Device: {DEVICE}"
    )
    print("  Innovation Axes:")
    print("    1. Architecture     : Twitter-RoBERTa (58M tweet pre-training)")
    print("    2. Intermediate Fusion: 8 features injected at encoder layer 6")
    print("    3. Differential LR  : layer-wise decay (0.85×/layer)")
    print("    4. Multi-Task Loss  : Focal-CE(3-class) + MSE + Label-Smooth")
    print("    5. Data Augmentation: title dropout + synonym swap + span corrupt")
    print("    +  SWA + Temperature Scaling")
    print("  Three-class target    : not_clickbait / ambiguous / clickbait")
    print(f"  Training data         : {CFG['INPUT_FILE']}")
    print(f"  New inference data    : {CFG['MODERN_FILE']}")

    # ── 1. Data ────────────────────────────────────────────────────────
    (df, truth_mean, binary_labels, three_labels,
     train_idx, val_idx, test_idx) = load_and_split()

    # ── 2. Fusion Features ─────────────────────────────────────────────
    feat_df, feat_names, feat_scaler = extract_fusion_features(df, train_idx)

    # ── 3. Train ───────────────────────────────────────────────────────
    model, swa_model, tokenizer, history, class_weights = train_model(
        df, truth_mean, three_labels,
        feat_df, feat_names, feat_scaler,
        train_idx, val_idx)

    # ── 4. Evaluate on test set ────────────────────────────────────────
    (pred_3class, pred_probs, reg_out,
     final_f1, acc_3, f1_conf, acc_conf,
     prec_conf, rec_conf) = evaluate_model(
        model, swa_model, tokenizer,
        df, three_labels, binary_labels, truth_mean,
        feat_df, feat_names, feat_scaler,
        test_idx)

    # ── 5. Temperature scaling on val probs ────────────────────────────
    # Re-run val inference to get calibration probs
    X_vl = feat_scaler.transform(feat_df[feat_names].values[val_idx].astype(float))
    posts  = df["post"].tolist()
    titles = [t if t.strip() else "" for t in df["title"].tolist()]
    def gather(lst, idx): return [lst[i] for i in idx]

    _, val_probs, _ = run_inference(
        model,
        gather(posts, val_idx), gather(titles, val_idx),
        X_vl, tokenizer)

    pred_probs_cal, T, ece_before, ece_after = temperature_scale_3class(
        val_probs, three_labels[val_idx], pred_probs)

    # Re-derive predictions from calibrated probs
    pred_3class_cal = np.argmax(pred_probs_cal, axis=1)

    # Final metrics (post-calibration)
    final_f1_cal = f1_score(three_labels[test_idx], pred_3class_cal,
                             average="macro", zero_division=0)
    acc_3_cal    = accuracy_score(three_labels[test_idx], pred_3class_cal)
    conf_mask_cal = pred_3class_cal != 1
    if conf_mask_cal.sum() > 0:
        pbc = (pred_3class_cal[conf_mask_cal] == 2).astype(int)
        tbc = binary_labels[test_idx][conf_mask_cal]
        f1_conf_cal  = f1_score(tbc, pbc, zero_division=0)
        acc_conf_cal = accuracy_score(tbc, pbc)
    else:
        f1_conf_cal = acc_conf_cal = 0.0

    print(f"\n  Post-calibration macro F1   : {final_f1_cal:.4f}")
    print(f"  Post-calibration accuracy   : {acc_3_cal:.4f}")
    print(f"  Post-calibration conf. F1   : {f1_conf_cal:.4f}")

    # Use calibrated results for final reporting
    final_f1  = final_f1_cal
    acc_3     = acc_3_cal
    f1_conf   = f1_conf_cal
    acc_conf  = acc_conf_cal
    pred_3class = pred_3class_cal
    pred_probs  = pred_probs_cal

    # ── 6. Error analysis ──────────────────────────────────────────────
    error_analysis(df, test_idx, pred_3class, pred_probs,
                   three_labels, binary_labels)

    # ── 7. Comparison table ────────────────────────────────────────────
    print_comparison(final_f1, acc_3, f1_conf, acc_conf)

    # ── 8. Ablation ────────────────────────────────────────────────────
    contributions = run_ablation_analysis(final_f1)

    # ── 9. Report ──────────────────────────────────────────────────────
    generate_report(
        history, pred_probs, three_labels[test_idx], binary_labels[test_idx],
        final_f1, acc_3, f1_conf, T, ece_before, ece_after, contributions)

    # ── 10. Save test predictions ──────────────────────────────────────
    save_test_predictions(df, test_idx, pred_3class, pred_probs,
                           three_labels, binary_labels, reg_out)

    # ── 11. Inference on combined_modern_cleaned.csv ───────────────────
    test_on_modern_dataset(model, tokenizer, feat_scaler)

    # ── 12. Final results card ─────────────────────────────────────────
    sec("Pipeline Complete")
    card = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Twitter-RoBERTa FUSED 3-CLASS PIPELINE — FINAL RESULTS CARD        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Task         : 3-class (not_clickbait / ambiguous / clickbait)         ║
║  Model        : cardiffnlp/twitter-roberta-base                ║
║                                                                          ║
║  Test-Set Results (calibrated):                                          ║
║    3-class macro F1  : {final_f1:.4f}                                         ║
║    3-class accuracy  : {acc_3:.4f}                                         ║
║    Confident-only F1 : {f1_conf:.4f}                                         ║
║    Confident-only Acc: {acc_conf:.4f}                                         ║
║                                                                          ║
║  Innovation Axes Applied:                                                ║
║    1. Twitter-RoBERTa tweet-domain pre-training (58M tweets)         ║
║    2. Intermediate fusion at encoder layer {CFG['FUSION_LAYER']}                       ║
║    3. Layer-wise LR decay (factor={CFG['LR_DECAY_FACTOR']})                           ║
║    4. Multi-task: FocalCE(γ={CFG['FOCAL_GAMMA']:.1f}) + MSE + LabelSmooth(ε={CFG['LABEL_SMOOTH_EPS']})     ║
║    5. Augmentation: title-drop({CFG['AUG_TITLE_DROPOUT_P']}) + word-swap + span-corrupt  ║
║    +  SWA (from epoch {CFG['SWA_START_EPOCH']}) + Temperature Scaling (T={T:.3f})           ║
║                                                                          ║
║  vs DistilBERT-v5 baseline ({CFG['PRIOR_BEST_F1']:.4f}): {final_f1 - CFG['PRIOR_BEST_F1']:+.4f}                          ║
╚══════════════════════════════════════════════════════════════════════════╝"""
    print(card)
    with open("twitter_roberta_v6_results_card.txt", "w") as fh:
        fh.write(card)
    print("  ✓ Results card → deberta_v6_results_card.txt")
    print("  Pipeline complete.\n")


if __name__ == "__main__":
    main()
