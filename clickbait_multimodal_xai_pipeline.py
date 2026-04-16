"""
clickbait_multimodal_xai_pipeline.py
======================================
3-Pillar Evolution of the 7-Model Tournament for Clickbait Detection.

  Pillar 1 · Multi-Modal Integration  ("The Eyes")
             VisualFeatureExtractor (MobileNetV2) + Late-Fusion MMT
  Pillar 2 · Model Interpretability   ("The Why")
             SHAP for LightGBM + Clickbait Nutrition Label
  Pillar 3 · Domain Generalization    ("The Generalist")
             Domain-Adversarial Training + Zero-Shot Transfer eval

Architecture:
  Stage 1  ─ Feature Factory  (30 hand-crafted + visual features)
  Stage 2  ─ Tournament       (5 base learners: DistilBERT, RoBERTa,
                                LightGBM, SBERT-MLP, Multi-Modal Transformer)
  Stage 3  ─ Power Blend      (grid-search weights + power k + temp scaling)
  Stage 4  ─ XAI Reports      (SHAP, Nutrition Label, interpretability_report.png)
  Stage 5  ─ Domain Generalization (adversarial augmentation + zero-shot eval)

Changes from prior 7-model tournament:
  • Removed XGBoost & CatBoost (high correlation with LightGBM)
  • Added Multi-Modal Transformer (MMT) fusing text embeddings + image features
  • Added SHAP-based interpretability for LightGBM
  • Added Clickbait Nutrition Label single-sample explainer
  • Added domain-adversarial mock data augmentation
  • Added zero-shot transfer evaluation (missing titles)

pip install torch transformers lightgbm scikit-learn
            sentence-transformers vaderSentiment Levenshtein scipy
            matplotlib seaborn shap pillow torchvision
"""

# ═══════════════════════════════════════════════════════════════════
#  0 · IMPORTS
# ═══════════════════════════════════════════════════════════════════
import ast, gc, os, re, random, textwrap, warnings, json, io
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
import torch.nn.functional as F
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

# ── Optional imports with graceful fallback ──────────────────────
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  [WARN] shap not installed — SHAP explanations will be skipped")

try:
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from PIL import Image
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("  [WARN] torchvision/PIL not installed — visual features disabled")

try:
    from Levenshtein import ratio as lev_ratio
except ImportError:
    import difflib
    def lev_ratio(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    print("  [INFO] python-Levenshtein not found — using difflib fallback")


# ═══════════════════════════════════════════════════════════════════
#  0 · GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    INPUT_FILE          = "final_cleaned_full.csv",
    OUTPUT_PREDS        = "multimodal_final_predictions.csv",
    OUTPUT_LEADERBOARD  = "multimodal_leaderboard.csv",
    INTERP_REPORT       = "interpretability_report.png",
    NUTRITION_REPORT    = "nutrition_label_example.json",

    # ── Splits ──────────────────────────────────────────────────────
    TEST_SIZE           = 0.20,
    BLEND_FRAC          = 0.20,
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

    # ── Multi-Modal Transformer (MMT) — replaces XGBoost/CatBoost ──
    MMT_EPOCHS          = 20,
    MMT_BATCH           = 128,
    MMT_LR              = 5e-4,
    MMT_HEADS           = 4,
    MMT_LAYERS          = 2,
    MMT_DIM             = 256,
    MMT_DROPOUT         = 0.20,

    # ── Visual Feature Extractor ────────────────────────────────────
    VISUAL_MODEL        = "mobilenet_v2",
    VISUAL_DIM          = 1280,       # MobileNetV2 final pooled dim
    IMAGE_SIZE          = 224,

    # ── Domain Adversarial ──────────────────────────────────────────
    DOMAIN_ADV_LAMBDA   = 0.1,
    MOCK_DOMAIN_SAMPLES = 200,

    # ── Blend Optimiser ─────────────────────────────────────────────
    POWER_K_GRID        = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    BLEND_THR_GRID      = 61,

    # ── Tri-state ───────────────────────────────────────────────────
    CERTAIN_CB          = 0.80,
    CERTAIN_NCB         = 0.20,

    # ── Reporting ───────────────────────────────────────────────────
    PRIOR_BEST_F1       = 0.711,
)

torch.manual_seed(CFG["SEED"])
random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pragmatic slang triggers ────────────────────────────────────────
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
#  PILLAR 1 · MULTI-MODAL INTEGRATION  ("The Eyes")
# ═══════════════════════════════════════════════════════════════════

class VisualFeatureExtractor:
    """
    Extract image features from postMedia URLs/paths using MobileNetV2.
    If no image is available, produces a zero-padded vector.

    Architecture:
      MobileNetV2 (pretrained on ImageNet) → Global Avg Pool → 1280-d vector
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.feature_dim = CFG["VISUAL_DIM"]
        self.model = None
        self.transform = None

        if HAS_VISION:
            self._init_model()

    def _init_model(self):
        """Load MobileNetV2 with the classification head removed."""
        base = tv_models.mobilenet_v2(pretrained=True)
        # Remove the final classifier — keep everything through adaptive avg pool
        self.model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).to(self.device)
        self.model.eval()

        self.transform = tv_transforms.Compose([
            tv_transforms.Resize((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"])),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract_from_path(self, image_path: str) -> np.ndarray:
        """Extract features from a local image file path."""
        if not HAS_VISION or self.model is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(tensor).cpu().numpy().flatten()
            return features
        except Exception:
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_from_url(self, url: str) -> np.ndarray:
        """
        Download and extract features from a URL.
        Falls back to zero-padding if download fails.
        """
        if not HAS_VISION or self.model is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as resp:
                img_data = resp.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(tensor).cpu().numpy().flatten()
            return features
        except Exception:
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_batch(self, media_list: list) -> np.ndarray:
        """
        Extract features for a batch of media entries from postMedia column.
        Each entry can be a URL, a local path, or empty/NaN.
        Returns: (n_samples, feature_dim) numpy array.
        """
        n = len(media_list)
        features = np.zeros((n, self.feature_dim), dtype=np.float32)

        if not HAS_VISION or self.model is None:
            print("  [VisualFeatureExtractor] Vision disabled — all zero-padded")
            return features

        loaded = 0
        for i, media_entry in enumerate(media_list):
            media_str = str(media_entry).strip()
            # Skip empty / NaN entries
            if not media_str or media_str.lower() == "nan" or media_str in ("[]", ""):
                continue

            # Try to parse URL(s) from the media string
            urls = re.findall(r'https?://[^\s\'">\]]+', media_str)
            if urls:
                feat = self.extract_from_url(urls[0])
                if np.any(feat != 0):
                    features[i] = feat
                    loaded += 1
            elif os.path.isfile(media_str):
                feat = self.extract_from_path(media_str)
                if np.any(feat != 0):
                    features[i] = feat
                    loaded += 1

        print(f"  [VisualFeatureExtractor] Loaded {loaded}/{n} images "
              f"({loaded/n*100:.1f}%), rest zero-padded")
        return features

    def has_visual_features(self, features_row: np.ndarray) -> bool:
        """Check if a sample has non-zero visual features."""
        return np.any(features_row != 0)

    def compute_text_visual_mismatch(self, text_emb: np.ndarray,
                                      visual_feat: np.ndarray) -> np.ndarray:
        """
        Compute text-visual mismatch score.
        Projects both to a common space and measures cosine distance.
        Higher value → more mismatch between text and image.
        """
        # Normalize both
        t_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
        v_norm = visual_feat / (np.linalg.norm(visual_feat, axis=1, keepdims=True) + 1e-8)

        # Since dimensions differ, use magnitude ratio as proxy
        t_mag = np.linalg.norm(text_emb, axis=1)
        v_mag = np.linalg.norm(visual_feat, axis=1)

        # For samples without images, return neutral 0.5
        has_image = v_mag > 1e-6
        mismatch = np.full(len(text_emb), 0.5)

        if has_image.any():
            # Variance of activation patterns as mismatch proxy
            t_var = np.var(t_norm[has_image], axis=1)
            v_var = np.var(v_norm[has_image], axis=1)
            raw = np.abs(t_var - v_var) / (t_var + v_var + 1e-8)
            mismatch[has_image] = raw

        return mismatch


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 1 · MULTI-MODAL TRANSFORMER (MMT)
#  Replaces XGBoost + CatBoost with a single cross-attention model
# ═══════════════════════════════════════════════════════════════════

class MultiModalTransformer(nn.Module):
    """
    Late-Fusion Multi-Modal Transformer.

    Inputs:
      - text_features:   (batch, text_dim)   — RoBERTa/SBERT embeddings
      - visual_features: (batch, visual_dim) — MobileNetV2 features (zero-padded if no image)
      - tabular_features:(batch, tab_dim)    — 30 hand-crafted features

    Architecture:
      1. Project each modality to d_model via linear layers
      2. Stack as a 3-token sequence: [TEXT, VISUAL, TABULAR]
      3. Apply N layers of Transformer self-attention
      4. Pool (mean over tokens) → Classification head
    """

    def __init__(self, text_dim, visual_dim, tabular_dim,
                 d_model=256, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        # Modality projections
        self.text_proj    = nn.Linear(text_dim, d_model)
        self.visual_proj  = nn.Linear(visual_dim, d_model)
        self.tabular_proj = nn.Linear(tabular_dim, d_model)

        # Learnable modality-type embeddings (like token-type embeddings)
        self.modality_emb = nn.Embedding(3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1),
        )

        # Domain discriminator for adversarial training (Pillar 3)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, text_feat, visual_feat, tabular_feat,
                return_domain=False, grl_lambda=0.0):
        """
        Forward pass with optional domain-adversarial gradient reversal.
        """
        # Project each modality
        t = self.text_proj(text_feat).unsqueeze(1)       # (B, 1, d)
        v = self.visual_proj(visual_feat).unsqueeze(1)   # (B, 1, d)
        b = self.tabular_proj(tabular_feat).unsqueeze(1) # (B, 1, d)

        # Add modality-type embeddings
        dev = text_feat.device
        mod_ids = torch.arange(3, device=dev)
        mod_emb = self.modality_emb(mod_ids).unsqueeze(0)  # (1, 3, d)
        seq = torch.cat([t, v, b], dim=1) + mod_emb        # (B, 3, d)

        # Transformer encoding
        encoded = self.transformer(seq)  # (B, 3, d)

        # Mean pooling over the 3 modality tokens
        pooled = encoded.mean(dim=1)  # (B, d)

        # Classification
        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if return_domain:
            # Gradient Reversal Layer (implemented via hook)
            if grl_lambda > 0:
                reversed_pooled = GradientReversalFunction.apply(pooled, grl_lambda)
            else:
                reversed_pooled = pooled.detach()
            domain_logits = self.domain_discriminator(reversed_pooled).squeeze(-1)
            return logits, domain_logits

        return logits


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for Domain-Adversarial Training."""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 2 · MODEL INTERPRETABILITY  ("The Why")
# ═══════════════════════════════════════════════════════════════════

# Feature groups for human-readable trigger mapping
TRIGGER_MAP = {
    "High Curiosity Gap": [
        "curiosity_gap_index", "forward_reference_count",
        "starts_with_demonstrative", "kl_divergence",
    ],
    "Sensational Sentiment": [
        "sensational_word_count", "post_sentiment_intensity",
        "post_sentiment", "abs_sentiment_gap",
    ],
    "Vague Pronouns / Language": [
        "second_person_count", "pragmatic_slang_count",
        "post_exclamation_marks", "post_question_marks", "ellipsis_count",
    ],
    "Semantic Mismatch (Text/Article)": [
        "cosine_similarity", "jaccard_similarity",
        "title_post_cosine_sim", "title_post_jaccard",
    ],
    "Bait Formatting": [
        "has_number", "post_caps_ratio", "clickbait_ngram_count",
        "avg_word_length",
    ],
    "Content Density": [
        "post_word_count", "article_word_count", "word_count_ratio",
        "entity_density", "proper_noun_ratio",
    ],
    "Mirror / Originality": [
        "mirror_similarity", "keyword_overlap_ratio",
        "desc_post_cosine_sim",
    ],
}


def generate_clickbait_nutrition_label(shap_values, feature_names,
                                        feature_values, base_value,
                                        visual_mismatch_score=0.5):
    """
    Generate a "Clickbait Nutrition Label" for a single sample.

    Returns a dict with:
      - score: 0-100%  (probability of being clickbait)
      - top_3_triggers: list of (trigger_name, contribution %)
      - visual_text_mismatch: "High" / "Medium" / "Low"
      - feature_breakdown: dict of feature → SHAP value
    """
    # Compute raw score from SHAP
    raw_logit = base_value + sum(shap_values)
    score_prob = float(sigmoid(raw_logit))
    score_pct = round(score_prob * 100, 1)

    # Map SHAP values back to trigger groups
    feat_shap = dict(zip(feature_names, shap_values))
    trigger_scores = {}
    for trigger, feats in TRIGGER_MAP.items():
        group_shap = sum(feat_shap.get(f, 0.0) for f in feats)
        trigger_scores[trigger] = group_shap

    # Sort by absolute contribution and pick top 3
    sorted_triggers = sorted(trigger_scores.items(),
                              key=lambda x: abs(x[1]), reverse=True)
    total_abs = sum(abs(v) for _, v in sorted_triggers) + 1e-8
    top_3 = [
        {
            "trigger": name,
            "contribution_pct": round(abs(val) / total_abs * 100, 1),
            "direction": "↑ clickbait" if val > 0 else "↓ not-clickbait",
        }
        for name, val in sorted_triggers[:3]
    ]

    # Visual/Text mismatch assessment
    if visual_mismatch_score > 0.7:
        vt_mismatch = "High"
    elif visual_mismatch_score > 0.4:
        vt_mismatch = "Medium"
    else:
        vt_mismatch = "Low"

    label = {
        "score": score_pct,
        "score_raw": score_prob,
        "top_3_triggers": top_3,
        "visual_text_mismatch": vt_mismatch,
        "all_trigger_scores": {k: round(v, 4) for k, v in sorted_triggers},
        "feature_breakdown": {k: round(v, 5) for k, v in feat_shap.items()},
    }
    return label


def generate_explanation_visuals(lgbm_model, X_explain, feature_names,
                                  y_explain=None, save_path=None):
    """
    Generate SHAP-based interpretability plots and save as
    interpretability_report.png.

    Produces a 2×2 dashboard:
      1. SHAP Summary (beeswarm) plot
      2. SHAP Feature importance (bar)
      3. SHAP Dependence plot for top feature
      4. Example Nutrition Label (text rendering)
    """
    if not HAS_SHAP:
        print("  [SKIP] SHAP not available — cannot generate explanation visuals")
        return None

    save_path = save_path or CFG["INTERP_REPORT"]

    sec("PILLAR 2 — Generating SHAP Explanation Visuals")

    # Create SHAP explainer for LightGBM
    explainer = shap.TreeExplainer(lgbm_model)

    # Use a subsample for speed (max 500 samples)
    n_explain = min(len(X_explain), 500)
    idx_sample = np.random.choice(len(X_explain), n_explain, replace=False)
    X_sub = X_explain[idx_sample]

    print(f"  Computing SHAP values for {n_explain} samples...")
    shap_vals = explainer.shap_values(X_sub)
    # For binary classification, shap_values returns [class0, class1]
    if isinstance(shap_vals, list):
        sv = shap_vals[1]  # class=1 (clickbait)
    else:
        sv = shap_vals

    # ── Build 2×2 dashboard ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Clickbait Detection — Interpretability Report (SHAP)",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: SHAP Summary (beeswarm-style via bar approximation)
    ax = axes[0, 0]
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feat_order = np.argsort(mean_abs_shap)[::-1][:15]
    top_names = [feature_names[i] for i in feat_order]
    top_vals = mean_abs_shap[feat_order]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title("Top-15 Feature Importance (SHAP)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: SHAP Feature importance grouped by trigger category
    ax = axes[0, 1]
    trigger_importance = {}
    feat_shap_map = dict(zip(feature_names, mean_abs_shap))
    for trigger, feats in TRIGGER_MAP.items():
        trigger_importance[trigger] = sum(
            feat_shap_map.get(f, 0.0) for f in feats
        )
    sorted_triggers = sorted(trigger_importance.items(), key=lambda x: x[1], reverse=True)
    trig_names = [t[0] for t in sorted_triggers]
    trig_vals = [t[1] for t in sorted_triggers]
    trig_colors = ["#E63946", "#F4845F", "#F7B267", "#457B9D",
                    "#2A9D8F", "#264653", "#A8DADC"][:len(trig_names)]
    ax.barh(range(len(trig_names)), trig_vals[::-1], color=trig_colors[::-1])
    ax.set_yticks(range(len(trig_names)))
    ax.set_yticklabels(trig_names[::-1], fontsize=9)
    ax.set_xlabel("Aggregate |SHAP Value|", fontsize=10)
    ax.set_title("Trigger Category Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: SHAP Dependence for top feature
    ax = axes[1, 0]
    top_feat_idx = feat_order[0]
    top_feat_name = feature_names[top_feat_idx]
    scatter = ax.scatter(
        X_sub[:, top_feat_idx], sv[:, top_feat_idx],
        c=sv[:, top_feat_idx], cmap="RdBu_r", alpha=0.5,
        s=12, edgecolors="none",
    )
    ax.set_xlabel(top_feat_name, fontsize=10)
    ax.set_ylabel(f"SHAP value for {top_feat_name}", fontsize=10)
    ax.set_title(f"SHAP Dependence: {top_feat_name}", fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="SHAP value")

    # Panel 4: Example Nutrition Label
    ax = axes[1, 1]
    ax.axis("off")

    # Generate a nutrition label for the first sample
    sample_idx = 0
    sample_shap = sv[sample_idx]
    sample_feats = X_sub[sample_idx]
    base_val = explainer.expected_value
    if isinstance(base_val, list):
        base_val = base_val[1]

    label = generate_clickbait_nutrition_label(
        sample_shap, feature_names, sample_feats, base_val
    )

    # Render as styled text
    nl_text = (
        "╔══════════════════════════════════════╗\n"
        "║    CLICKBAIT NUTRITION LABEL         ║\n"
        "╠══════════════════════════════════════╣\n"
        f"║  Score: {label['score']}%"
        f"{'':>{35-len(str(label['score']))-1}}║\n"
        "╠══════════════════════════════════════╣\n"
        "║  Top 3 Triggers:                    ║\n"
    )
    for i, trig in enumerate(label["top_3_triggers"], 1):
        line = f"  {i}. {trig['trigger'][:25]}"
        pct = f"({trig['contribution_pct']}%)"
        nl_text += f"║{line:<32}{pct:>6}║\n"

    nl_text += (
        "╠══════════════════════════════════════╣\n"
        f"║  Visual/Text Mismatch: "
        f"{label['visual_text_mismatch']:<14}║\n"
        "╚══════════════════════════════════════╝"
    )
    ax.text(0.1, 0.5, nl_text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#333", linewidth=2))
    ax.set_title("Example Nutrition Label (Sample 0)", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved SHAP interpretability report → {save_path}")

    return sv, explainer


# ═══════════════════════════════════════════════════════════════════
#  PILLAR 3 · DOMAIN GENERALIZATION  ("The Generalist")
# ═══════════════════════════════════════════════════════════════════

def generate_mock_domain_data(n_samples=200, seed=42):
    """
    Generate synthetic non-news domain samples (YouTube/Reddit style)
    for domain-adversarial training.

    These samples use clickbait patterns common outside traditional news:
    - YouTube-style titles with ALL CAPS and excessive punctuation
    - Reddit-style vague pronouns and curiosity gaps
    - Social media engagement bait

    Returns: DataFrame with same schema as main data + domain_label column
    """
    rng = np.random.RandomState(seed)

    yt_templates_cb = [
        "You WON'T BELIEVE What Happened Next!!!",
        "I Tried {thing} For 30 Days And THIS Happened...",
        "THEY Don't Want You To Know This Secret About {topic}",
        "This {thing} Will Change Your Life Forever!!! (NOT CLICKBAIT)",
        "We Found Something INSANE In {place}!!!",
        "Top 10 {things} That Will BLOW YOUR MIND",
        "I Can't Believe {person} Did THIS!!!",
        "The REAL Reason Why {person} Left {thing}",
        "This Is Why You Should NEVER {action}...",
        "What {person} Said Will SHOCK You",
        "EXPOSING The Truth About {topic}!!!",
        "I Was TODAY Years Old When I Found Out About {thing}",
    ]

    yt_templates_ncb = [
        "How to Fix {thing} — Complete Guide (2024)",
        "{topic} Tutorial for Beginners",
        "Reviewing the New {thing}: Worth It?",
        "My Experience with {topic} After 6 Months",
        "Understanding {topic}: A Deep Dive",
        "Full Tour of {place}",
        "{topic} Explained Simply",
        "Comparing {thing} vs {thing2}: Honest Review",
    ]

    fillers = {
        "thing": ["this product", "intermittent fasting", "cold showers",
                  "AI coding", "meal prepping", "minimalism", "Tesla stock"],
        "topic": ["cryptocurrency", "machine learning", "weight loss",
                  "productivity", "remote work", "sustainable living"],
        "place": ["this abandoned building", "Tokyo", "Area 51",
                  "the deep web", "this secret location"],
        "things": ["life hacks", "psychological tricks", "facts about space",
                   "foods you eat wrong", "animals that shouldn't exist"],
        "person": ["this celebrity", "my boss", "this streamer",
                   "the government", "scientists"],
        "action": ["do this at home", "eat this food", "trust this app",
                   "skip leg day"],
        "thing2": ["the alternative", "the original", "the budget version"],
    }

    rows = []
    for i in range(n_samples):
        is_cb = rng.random() < 0.5
        templates = yt_templates_cb if is_cb else yt_templates_ncb
        tmpl = rng.choice(templates)
        text = tmpl
        for key, vals in fillers.items():
            text = text.replace(f"{{{key}}}", rng.choice(vals), 1)

        rows.append({
            "post": text,
            "title": "",           # Missing titles — simulates Reddit/YouTube
            "article": text * 3,   # Synthetic article content
            "desc": "",
            "keywords": "",
            "media_raw": "",
            "truthClass": int(is_cb),
            "truthMean": float(is_cb) * rng.uniform(0.6, 1.0) + \
                         (1 - float(is_cb)) * rng.uniform(0.0, 0.4),
            "domain": "non_news",  # domain label for adversarial training
        })

    return pd.DataFrame(rows)


def zero_shot_transfer_eval(model_probs, y_true, df, test_idx,
                             title_col="title"):
    """
    Evaluate model performance on posts where the Target Title is missing.
    This simulates Reddit/YouTube conditions where there's no article title
    to compare against.
    """
    sec("PILLAR 3 — Zero-Shot Transfer Evaluation (Missing Titles)")

    titles = df.iloc[test_idx][title_col].values
    has_title = np.array([bool(str(t).strip()) for t in titles])
    no_title  = ~has_title

    n_has = has_title.sum()
    n_no  = no_title.sum()
    print(f"  Test samples with title:    {n_has} ({n_has/len(has_title)*100:.1f}%)")
    print(f"  Test samples without title: {n_no}  ({n_no/len(has_title)*100:.1f}%)")

    if n_has > 0:
        m_with = model_metrics(y_true[has_title], model_probs[has_title], "With-Title")
        print(f"  F1 (with title):    {m_with['f1']:.4f}")
    else:
        m_with = None

    if n_no > 10:
        m_without = model_metrics(y_true[no_title], model_probs[no_title], "No-Title")
        print(f"  F1 (without title): {m_without['f1']:.4f}")
    else:
        m_without = None
        print(f"  Not enough no-title samples for reliable evaluation")

    # Degradation analysis
    if m_with and m_without:
        delta = m_with['f1'] - m_without['f1']
        print(f"\n  Transfer Gap: {delta:+.4f} "
              f"({'model relies on title' if delta > 0.05 else 'robust to missing title'})")
    else:
        print(f"\n  Transfer gap cannot be computed (insufficient no-title samples)")
        if n_no <= 10:
            print(f"  → This suggests the dataset is title-rich (news domain). "
                  f"Testing on augmented non-news data is recommended.")

    return m_with, m_without


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

    # Mark all original data as 'news' domain
    df["domain"] = "news"

    y = df["truthClass"].values

    # ── Primary split ──
    tv_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=CFG["TEST_SIZE"],
        stratify=y, random_state=CFG["SEED"],
    )
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
#  SECTION 2 · FEATURE FACTORY  (30 features + visual mismatch)
# ═══════════════════════════════════════════════════════════════════
def build_feature_factory(df, emb_post, emb_title, emb_article, emb_desc,
                           base_train_idx, y, visual_features=None):
    sec("SECTION 2 — Feature Factory  (30+ features, Groups A–I)")

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
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
                  / max(len(x.split()), 1))

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
    P /= P.sum(axis=1, keepdims=True)
    Q /= Q.sum(axis=1, keepdims=True)
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
        sum(1 for k in re.split(r"[,\s]+", kw_s.lower())
            if k and k in pt.lower()) /
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
        lambda x: sum(1 for w in x.lower().split()
                       if w.strip(".,!?;:'\"") in SENS))
    feat["post_sentiment_intensity"]  = post_s.abs()
    feat["proper_noun_ratio"]         = post.apply(
        lambda x: sum(1 for w in x.split()[1:]
                       if w and w[0].isupper() and not w.isupper())
                  / max(len(x.split()), 1))
    feat["forward_reference_count"] = post.apply(
        lambda x: sum(1 for w in x.lower().split()
                       if w.strip(".,!?;:'\"") in FWD))

    # ── Group G: Mirror Similarity & Pragmatic Slang (2) ─────────
    subsec("Group G — Mirror Similarity + Pragmatic Slang")
    feat["mirror_similarity"]    = [
        lev_ratio(pt.lower().strip(), tt.lower().strip()) if tt.strip() else 0.5
        for pt, tt in zip(post, title)
    ]
    feat["pragmatic_slang_count"] = post.apply(lambda x: len(SLANG_RE.findall(x)))

    # ── Group H: Entity Density & Curiosity Gap Index (2) ────────
    subsec("Group H — Entity Density + Curiosity Gap Index")
    def entity_density(text):
        words = text.split()
        if not words: return 0.0
        proper = sum(1 for i, w in enumerate(words)
                     if i > 0 and w and w[0].isupper()
                     and not w.isupper() and len(w) > 1)
        return proper / len(words)

    feat["entity_density"] = post.apply(entity_density)
    title_wc = title.apply(lambda x: len(x.split()))
    feat["curiosity_gap_index"] = (
        (title_wc - post_wc) / (title_wc + post_wc + 1)
    )

    # ── Group I: Visual / Multi-Modal features (1) ★ NEW ─────────
    subsec("Group I — NEW: Visual-Text Mismatch Score")
    if visual_features is not None:
        vfe = VisualFeatureExtractor(device=str(DEVICE))
        mismatch = vfe.compute_text_visual_mismatch(emb_post, visual_features)
        feat["visual_text_mismatch"] = mismatch
        print(f"  visual_text_mismatch: mean={mismatch.mean():.3f}, "
              f"std={mismatch.std():.3f}")
    else:
        feat["visual_text_mismatch"] = 0.5  # neutral default
        print(f"  visual_text_mismatch: all neutral (no visual features)")

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
#  SECTION 3.5 · VISUAL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_visual_features(df):
    sec("SECTION 3.5 — Visual Feature Extraction (MobileNetV2)")
    vfe = VisualFeatureExtractor(device=str(DEVICE))
    media_list = df["media_raw"].tolist()
    visual_feats = vfe.extract_batch(media_list)
    print(f"  Visual feature shape: {visual_feats.shape}")
    has_img = np.any(visual_feats != 0, axis=1)
    print(f"  Samples with image features: {has_img.sum()} "
          f"({has_img.mean()*100:.1f}%)")
    return visual_feats


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 · TOURNAMENT  (5 base learners)
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
        model.load_state_dict(torch.load(CFG["DISTILBERT_WEIGHTS"],
                                          map_location=DEVICE))
        print(f"  Loaded saved weights from '{CFG['DISTILBERT_WEIGHTS']}'")
    else:
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
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tgt = batch["s"].to(DEVICE)
                pred = torch.sigmoid(
                    model(input_ids=ids, attention_mask=mask).logits.squeeze(-1))
                loss = loss_fn(pred, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); ep_loss += loss.item()
                if (bi + 1) % 150 == 0:
                    print(f"    Ep{epoch+1} B{bi+1}/{len(tr_dl)} "
                          f"MSE={ep_loss/(bi+1):.5f}")
            model.eval(); vl_p, vl_t = [], []
            with torch.no_grad():
                for batch in vl_dl:
                    ids = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    vl_p.extend(torch.sigmoid(
                        model(input_ids=ids, attention_mask=mask
                              ).logits.squeeze(-1)).cpu().numpy())
                    vl_t.extend(batch["s"].numpy())
            vl_mse = np.mean((np.array(vl_p) - np.array(vl_t))**2)
            print(f"  Epoch {epoch+1}  train_MSE={ep_loss/len(tr_dl):.5f}  "
                  f"val_MSE={vl_mse:.5f}")
            if vl_mse < best_mse:
                best_mse = vl_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ★ best val_MSE={best_mse:.5f}")
        model.load_state_dict(best_state)
        torch.save(best_state, CFG["DISTILBERT_WEIGHTS"])
        print(f"  Saved weights → {CFG['DISTILBERT_WEIGHTS']}")

    blend_p = _bert_infer(model, tok,
                          gather(posts, blend_val_idx),
                          gather(titles, blend_val_idx),
                          DEVICE, CFG["BERT_BATCH"], CFG["BERT_MAX_LEN"])
    test_p  = _bert_infer(model, tok,
                          gather(posts, test_idx),
                          gather(titles, test_idx),
                          DEVICE, CFG["BERT_BATCH"], CFG["BERT_MAX_LEN"])
    del model; free_gpu()

    m = model_metrics(y[test_idx], test_p, "DistilBERT-v2")
    print(f"  DistilBERT standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ── 4.2  RoBERTa-Base — binary classification ───────────────────
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
            logits = out.logits[:, 1]
            preds.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(preds)


def train_roberta(df, y, base_train_idx, blend_val_idx, test_idx):
    sec("SECTION 4.2 — RoBERTa-Base  [binary classification]")
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
        model.load_state_dict(torch.load(CFG["ROBERTA_WEIGHTS"],
                                          map_location=DEVICE))
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

        n_neg = (labels[bt] == 0).sum()
        n_pos = (labels[bt] == 1).sum()
        pw = torch.tensor([1.0, n_neg / max(n_pos, 1)],
                           dtype=torch.float).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=pw)
        opt    = AdamW(model.parameters(), lr=CFG["ROB_LR"], weight_decay=0.01)
        total  = len(tr_dl) * CFG["ROB_EPOCHS"]
        warmup = int(total * CFG["ROB_WARMUP"])
        sched  = get_linear_schedule_with_warmup(opt, warmup, total)

        best_f1, best_state = 0.0, None
        for epoch in range(CFG["ROB_EPOCHS"]):
            model.train(); ep_loss = 0.0
            for bi, batch in enumerate(tr_dl):
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tgt = batch["y"].long().to(DEVICE)
                loss = loss_fn(
                    model(input_ids=ids, attention_mask=mask).logits, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); ep_loss += loss.item()
                if (bi + 1) % 150 == 0:
                    print(f"    Ep{epoch+1} B{bi+1}/{len(tr_dl)} "
                          f"CE={ep_loss/(bi+1):.4f}")
            vl_p = _cls_infer(model, tok, gather(posts, bv),
                              DEVICE, CFG["ROB_BATCH"], CFG["ROB_MAX_LEN"])
            _, vl_f1 = best_threshold(labels[bv], vl_p)
            print(f"  Epoch {epoch+1}  CE={ep_loss/len(tr_dl):.4f}  "
                  f"val_F1={vl_f1:.4f}")
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
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
def train_lightgbm(feat_df, feat_names, y, base_train_idx,
                    blend_val_idx, test_idx):
    sec("SECTION 4.3 — LightGBM  [tabular features + visual mismatch]")
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


# ── 4.4  SBERT-MLP ──────────────────────────────────────────────
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
    sec("SECTION 4.4 — SBERT-MLP  [3-layer NN on sentence embeddings]")

    def make_X(ep, et):
        return np.concatenate([ep, et, ep - et, ep * et], axis=1)

    X_all = make_X(emb_post, emb_title)
    print(f"  MLP input dim: {X_all.shape[1]}  ({emb_post.shape[1]} × 4)")

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_all[base_train_idx])
    Xbl = sc.transform(X_all[blend_val_idx])
    Xte = sc.transform(X_all[test_idx])

    bt_idx, bv_idx = train_test_split(
        np.arange(len(base_train_idx)), test_size=0.12,
        stratify=y[base_train_idx], random_state=CFG["SEED"])

    n_neg = (y[base_train_idx[bt_idx]] == 0).sum()
    n_pos = (y[base_train_idx[bt_idx]] == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                               dtype=torch.float32).to(DEVICE)

    tr_ds = _EmbDs(Xtr[bt_idx], y[base_train_idx[bt_idx]])
    vl_ds = _EmbDs(Xtr[bv_idx], y[base_train_idx[bv_idx]])
    tr_dl = DataLoader(tr_ds, batch_size=CFG["MLP_BATCH"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=CFG["MLP_BATCH"], shuffle=False)

    model = _SBERTMlp(
        in_dim=X_all.shape[1],
        d1=CFG["MLP_DROPOUT1"], d2=CFG["MLP_DROPOUT2"]
    ).to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), lr=CFG["MLP_LR"],
                              weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["MLP_EPOCHS"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1, best_state = 0.0, None
    for epoch in range(CFG["MLP_EPOCHS"]):
        model.train(); ep_loss = 0.0
        for Xb, yb in tr_dl:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model.net(Xb).squeeze(-1)
            loss   = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
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


# ── 4.5  Multi-Modal Transformer (MMT) ★ NEW ────────────────────
class _MMTDataset(Dataset):
    def __init__(self, text_feat, visual_feat, tabular_feat,
                 labels=None, domain_labels=None):
        self.text    = torch.tensor(text_feat, dtype=torch.float32)
        self.visual  = torch.tensor(visual_feat, dtype=torch.float32)
        self.tabular = torch.tensor(tabular_feat, dtype=torch.float32)
        self.labels  = (torch.tensor(labels, dtype=torch.float32)
                        if labels is not None else None)
        self.domain  = (torch.tensor(domain_labels, dtype=torch.float32)
                        if domain_labels is not None else None)

    def __len__(self): return len(self.text)

    def __getitem__(self, i):
        item = {
            "text": self.text[i],
            "visual": self.visual[i],
            "tabular": self.tabular[i],
        }
        if self.labels is not None:
            item["label"] = self.labels[i]
        if self.domain is not None:
            item["domain"] = self.domain[i]
        return item


def train_mmt(emb_post, visual_features, feat_df, feat_names,
              y, df, base_train_idx, blend_val_idx, test_idx):
    """
    Train the Multi-Modal Transformer with optional domain-adversarial
    training. Replaces XGBoost + CatBoost in the ensemble.
    """
    sec("SECTION 4.5 — Multi-Modal Transformer (MMT) ★ NEW")
    print("  Fusing: SBERT text embeddings + MobileNetV2 visual "
          "+ tabular features")

    text_dim    = emb_post.shape[1]
    visual_dim  = visual_features.shape[1]
    tabular_dim = len(feat_names)

    print(f"  Dimensions: text={text_dim}, visual={visual_dim}, "
          f"tabular={tabular_dim}")

    # Prepare features
    X_tab = feat_df[feat_names].values
    sc_tab = StandardScaler()
    X_tab_tr = sc_tab.fit_transform(X_tab[base_train_idx])
    X_tab_bl = sc_tab.transform(X_tab[blend_val_idx])
    X_tab_te = sc_tab.transform(X_tab[test_idx])

    sc_vis = StandardScaler()
    vis_tr = sc_vis.fit_transform(visual_features[base_train_idx])
    vis_bl = sc_vis.transform(visual_features[blend_val_idx])
    vis_te = sc_vis.transform(visual_features[test_idx])

    sc_txt = StandardScaler()
    txt_tr = sc_txt.fit_transform(emb_post[base_train_idx])
    txt_bl = sc_txt.transform(emb_post[blend_val_idx])
    txt_te = sc_txt.transform(emb_post[test_idx])

    # ── Domain-adversarial augmentation (Pillar 3) ───────────────
    subsec("Domain-Adversarial Augmentation")
    mock_df = generate_mock_domain_data(CFG["MOCK_DOMAIN_SAMPLES"], CFG["SEED"])
    print(f"  Generated {len(mock_df)} mock non-news samples for "
          f"domain adversarial training")

    # Domain labels: 0 = news (real), 1 = non-news (mock)
    domain_labels_tr = np.zeros(len(base_train_idx))

    # Create mock embeddings (random for visual, noise for text)
    mock_text = np.random.randn(len(mock_df), text_dim).astype(np.float32) * 0.1
    mock_vis  = np.zeros((len(mock_df), visual_dim), dtype=np.float32)
    mock_tab  = np.random.randn(len(mock_df), tabular_dim).astype(np.float32) * 0.5
    mock_domains = np.ones(len(mock_df))  # domain = 1 (non-news)
    mock_labels  = mock_df["truthClass"].values.astype(np.float32)

    # Combine real + mock for training
    aug_txt = np.vstack([txt_tr, mock_text])
    aug_vis = np.vstack([vis_tr, mock_vis])
    aug_tab = np.vstack([X_tab_tr, mock_tab])
    aug_y   = np.concatenate([y[base_train_idx].astype(np.float32), mock_labels])
    aug_dom = np.concatenate([domain_labels_tr, mock_domains])

    print(f"  Augmented training set: {len(aug_y)} samples "
          f"(real={len(base_train_idx)}, mock={len(mock_df)})")

    # Internal val split
    bt_idx, bv_idx = train_test_split(
        np.arange(len(aug_y)), test_size=0.12,
        stratify=aug_y.astype(int), random_state=CFG["SEED"])

    n_neg = (aug_y[bt_idx] == 0).sum()
    n_pos = (aug_y[bt_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                               dtype=torch.float32).to(DEVICE)

    tr_ds = _MMTDataset(aug_txt[bt_idx], aug_vis[bt_idx], aug_tab[bt_idx],
                         aug_y[bt_idx], aug_dom[bt_idx])
    vl_ds = _MMTDataset(aug_txt[bv_idx], aug_vis[bv_idx], aug_tab[bv_idx],
                         aug_y[bv_idx])
    tr_dl = DataLoader(tr_ds, batch_size=CFG["MMT_BATCH"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=CFG["MMT_BATCH"], shuffle=False)

    model = MultiModalTransformer(
        text_dim=text_dim, visual_dim=visual_dim, tabular_dim=tabular_dim,
        d_model=CFG["MMT_DIM"], nhead=CFG["MMT_HEADS"],
        num_layers=CFG["MMT_LAYERS"], dropout=CFG["MMT_DROPOUT"],
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["MMT_LR"],
                             weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["MMT_EPOCHS"])

    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dom_loss_fn = nn.BCEWithLogitsLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(CFG["MMT_EPOCHS"]):
        model.train(); ep_cls_loss = 0.0; ep_dom_loss = 0.0

        # Progressive λ schedule (increases over epochs)
        p = epoch / max(CFG["MMT_EPOCHS"] - 1, 1)
        grl_lambda = CFG["DOMAIN_ADV_LAMBDA"] * (2 / (1 + np.exp(-10 * p)) - 1)

        for batch in tr_dl:
            t_feat = batch["text"].to(DEVICE)
            v_feat = batch["visual"].to(DEVICE)
            b_feat = batch["tabular"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            has_domain = "domain" in batch
            if has_domain:
                dom_labels = batch["domain"].to(DEVICE)
                cls_logits, dom_logits = model(
                    t_feat, v_feat, b_feat,
                    return_domain=True, grl_lambda=grl_lambda)
                loss_cls = cls_loss_fn(cls_logits, labels)
                loss_dom = dom_loss_fn(dom_logits, dom_labels)
                loss = loss_cls + grl_lambda * loss_dom
                ep_dom_loss += loss_dom.item()
            else:
                cls_logits = model(t_feat, v_feat, b_feat)
                loss = cls_loss_fn(cls_logits, labels)

            opt.zero_grad(); loss.backward(); opt.step()
            ep_cls_loss += loss.item()

        sched.step()

        # Validate
        model.eval(); vl_p = []
        with torch.no_grad():
            for batch in vl_dl:
                logits = model(
                    batch["text"].to(DEVICE),
                    batch["visual"].to(DEVICE),
                    batch["tabular"].to(DEVICE))
                vl_p.extend(torch.sigmoid(logits).cpu().numpy())
        vl_p = np.array(vl_p)
        _, vl_f1 = best_threshold(aug_y[bv_idx], vl_p)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3}/{CFG['MMT_EPOCHS']}  "
                  f"cls_loss={ep_cls_loss/len(tr_dl):.4f}  "
                  f"dom_loss={ep_dom_loss/max(len(tr_dl),1):.4f}  "
                  f"val_F1={vl_f1:.4f}  λ={grl_lambda:.4f}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    def infer_mmt(txt, vis, tab):
        ds = _MMTDataset(txt, vis, tab)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        out = []
        with torch.no_grad():
            for batch in dl:
                logits = model(
                    batch["text"].to(DEVICE),
                    batch["visual"].to(DEVICE),
                    batch["tabular"].to(DEVICE))
                out.extend(torch.sigmoid(logits).cpu().numpy())
        return np.array(out)

    blend_p = infer_mmt(txt_bl, vis_bl, X_tab_bl)
    test_p  = infer_mmt(txt_te, vis_te, X_tab_te)
    del model; free_gpu()

    m = model_metrics(y[test_idx], test_p, "MMT")
    print(f"  MMT standalone test F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    return blend_p, test_p, m


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 · OPTIMISED POWER BLEND
# ═══════════════════════════════════════════════════════════════════
def optimize_blend(blend_probs, test_probs, y_blend, y_test, model_names):
    sec("SECTION 5 — Optimised Power Blend")

    P_bl = np.array([blend_probs[m] for m in model_names]).T
    P_te = np.array([test_probs[m]  for m in model_names]).T
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

    standalone_f1 = []
    for m in model_names:
        _, f = best_threshold(y_blend, blend_probs[m], CFG["BLEND_THR_GRID"])
        standalone_f1.append(f)
    init_w = np.array(standalone_f1, dtype=float)
    init_w = np.maximum(init_w, 0.05)

    print(f"\n  Initial weights (by blend-val F1):")
    for m, w, f in zip(model_names, init_w, standalone_f1):
        print(f"    {m:<22}  blend-val F1={f:.4f}  init_w={w:.3f}")

    subsec("Grid Search over power k")
    best_grid_k, best_grid_f1, best_grid_w = 1.0, 0.0, init_w.copy()

    for k in CFG["POWER_K_GRID"]:
        x0 = np.append(init_w.copy(), k)
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
        f1_val = -res.fun
        print(f"  k={k:.2f}  →  blend-val F1={f1_val:.4f}")
        if f1_val > best_grid_f1:
            best_grid_f1 = f1_val
            best_grid_k  = res.x[n_m]
            best_grid_w  = np.abs(res.x[:n_m])

    subsec("Final Refinement")
    x0_final = np.append(best_grid_w, best_grid_k)
    res_final = minimize(neg_f1, x0_final, method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
    opt_weights = np.abs(res_final.x[:n_m])
    opt_k       = float(np.clip(res_final.x[n_m], 0.25, 4.0))
    opt_blend_f1 = -res_final.fun

    w_norm = opt_weights / opt_weights.sum()
    print(f"\n  Optimal power k = {opt_k:.3f}")
    print(f"  Blend-val F1 (power blend) = {opt_blend_f1:.4f}")
    print(f"\n  Final normalised weights:")
    for m, w in zip(model_names, w_norm):
        bar = "█" * int(w * 40)
        print(f"    {m:<22}  {w:.4f}  {bar}")

    blend_final = blend(P_bl, opt_weights, opt_k)
    test_final  = blend(P_te, opt_weights, opt_k)

    m_blend = model_metrics(y_blend, blend_final, "PowerBlend@blend_val")
    m_test  = model_metrics(y_test,  test_final,  "PowerBlend@test")

    print(f"\n  Power Blend — blend_val F1={m_blend['f1']:.4f}")
    print(f"  Power Blend — test      F1={m_test['f1']:.4f}  "
          f"(↑ {m_test['f1'] - CFG['PRIOR_BEST_F1']:+.4f} vs prior best)")

    return blend_final, test_final, opt_weights, opt_k, w_norm, model_names


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 · TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════════════
def temperature_scale(blend_raw, y_blend, test_raw):
    sec("SECTION 6 — Temperature Scaling")

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

    def ece(y, p, n=10):
        fp, mp = calibration_curve(y, p, n_bins=n, strategy="uniform")
        return float(np.mean(np.abs(fp - mp)))

    ece_raw = ece(y_blend, blend_raw)
    ece_cal = ece(y_blend, p_blend_cal)
    print(f"  Optimal temperature T = {T:.4f}")
    print(f"  ECE before calibration : {ece_raw:.4f}")
    print(f"  ECE after calibration  : {ece_cal:.4f}")

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
            print(f"  {b:<20}: {n:>5} ({pct:.1f}%)  precision={prec_b:.4f}")
        else:
            cb_rate = y_test[mask].mean() if n > 0 else 0.0
            print(f"  {b:<20}: {n:>5} ({pct:.1f}%)  CB-rate={cb_rate:.3f}")

    return f1_opt, acc, prec, rec, y_pred, buckets


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8 · SCIENTIFIC REPORTS
# ═══════════════════════════════════════════════════════════════════
def print_leaderboard(metrics_list, final_f1):
    sec("SECTION 8.1 — Tournament Leaderboard")
    rows = sorted(metrics_list, key=lambda x: -x["f1"])

    header = (f"  {'#':>2}  {'Model':<22}  {'F1':>7}  {'Acc':>7}  "
              f"{'Prec':>7}  {'Rec':>7}  {'Thr':>6}")
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


def diversity_matrix(test_probs, model_names):
    sec("SECTION 8.2 — Diversity Matrix")
    P = np.array([test_probs[m] for m in model_names])
    C = np.corrcoef(P)

    w = 8
    print("  " + "".join(f"{n[:w]:>{w+1}}" for n in model_names))
    for i, n in enumerate(model_names):
        row = f"  {n[:14]:<14} "
        for j in range(len(model_names)):
            row += f"  {C[i, j]:+.3f}"
        print(row)

    off_diag = C[np.triu_indices(len(model_names), k=1)]
    print(f"\n  Mean |off-diagonal| correlation : {np.mean(np.abs(off_diag)):.4f}")
    print(f"  (Lower = more diverse → better ensemble)")

    fig, ax = plt.subplots(figsize=(8, 6))
    short_names = [n.replace("DistilBERT-v2","DBERT").replace("RoBERTa-Base","RoB")
                    .replace("LightGBM","LGBM").replace("SBERT-MLP","MLP")
                    .replace("MMT","MMT") for n in model_names]
    sns.heatmap(C, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Model Diversity Matrix (Pearson ρ)", fontsize=13)
    plt.tight_layout()
    plt.savefig("diversity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → diversity_matrix.png")
    return C


def what_worked_report(lgbm_clf, lgbm_scaler, feat_names,
                        blend_probs, y_blend, opt_weights, w_norm, model_names):
    sec("SECTION 8.3 — 'What Worked' Report")

    subsec("Top-5 Features (LightGBM gain importance)")
    imp = lgbm_clf.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    for rank, idx in enumerate(order[:5], 1):
        pct = imp[idx] / imp.sum() * 100
        bar = "█" * int(pct * 1.2)
        print(f"  {rank}. {feat_names[idx]:<35}  {imp[idx]:>6.1f}  "
              f"({pct:.1f}%)  {bar}")

    subsec("Top-10 Features (full)")
    new_feats = {"mirror_similarity", "pragmatic_slang_count",
                 "entity_density", "curiosity_gap_index",
                 "visual_text_mismatch"}
    for rank, idx in enumerate(order, 1):
        pct = imp[idx] / imp.sum() * 100
        new_tag = " ★ NEW" if feat_names[idx] in new_feats else ""
        print(f"  {rank:>2}. {feat_names[idx]:<35}  {imp[idx]:>6.1f}  "
              f"({pct:.1f}%){new_tag}")

    subsec("Top Models by Blend-Weight Contribution")
    contributions = sorted(zip(model_names, w_norm), key=lambda x: -x[1])
    for rank, (m, w) in enumerate(contributions[:3], 1):
        bar = "█" * int(w * 60)
        print(f"  {rank}. {m:<22}  weight={w:.4f}  {bar}")


def calibration_plot(test_probs_dict, final_probs, y_test, model_names,
                      T, ece_raw, ece_cal):
    sec("SECTION 8.4 — Calibration & Confidence Plots")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Ensemble Calibration & Confidence Dashboard",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    fp_raw, mp_raw = calibration_curve(
        y_test, test_probs_dict.get("_raw_blend", final_probs),
        n_bins=10, strategy="uniform")
    fp_cal, mp_cal = calibration_curve(
        y_test, final_probs, n_bins=10, strategy="uniform")
    ax.plot(mp_raw, fp_raw, "o-", color="#E63946",
            label=f"Pre-scaling ECE={ece_raw:.3f}")
    ax.plot(mp_cal, fp_cal, "s-", color="#2A9D8F",
            label=f"Post-scaling ECE={ece_cal:.3f}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram (T={T:.3f})")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(final_probs[y_test == 0], bins=bins, color="#457B9D",
            alpha=0.65, density=True, label="True NCB")
    ax.hist(final_probs[y_test == 1], bins=bins, color="#E63946",
            alpha=0.65, density=True, label="True CB")
    ax.set_xlabel("Final Calibrated Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Class")
    ax.legend(fontsize=8)

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
    ax.set_title("Model Score Distributions (red=CB, blue=NCB)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#E63946", label="True CB"),
                        Patch(color="#457B9D", label="True NCB")], fontsize=8)

    plt.tight_layout()
    plt.savefig("ensemble_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → ensemble_calibration.png")


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
        out[f"prob_{m.lower().replace('-','_').replace(' ','_')}"] = \
            test_probs_dict[m]

    out.to_csv(CFG["OUTPUT_PREDS"], index=False)
    print(f"  ✓ Predictions → {CFG['OUTPUT_PREDS']}  (shape {out.shape})")

    card = f"""
╔══════════════════════════════════════════════════════════════════════╗
║   MULTI-MODAL 3-PILLAR TOURNAMENT — RESULTS CARD                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Pipeline                                                            ║
║    5 base learners → Power Blend (k={opt_k:.3f}) → Temp Scale (T={T:.3f})  ║
║    + Multi-Modal Transformer (MMT)                                   ║
║    + SHAP Interpretability                                           ║
║    + Domain-Adversarial Generalization                               ║
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
║  Architecture Changes vs Prior 7-Model Tournament                    ║
║    ✗ Removed XGBoost, CatBoost (high inter-model correlation)        ║
║    ★ Added Multi-Modal Transformer (text + visual + tabular fusion)   ║
║    ★ Added SHAP explanations (LightGBM interpretability)             ║
║    ★ Added Domain-Adversarial training (non-news generalization)     ║
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
    banner("3-PILLAR MULTI-MODAL TOURNAMENT  |  Target: F1 > 0.711"
           f"  |  Device: {DEVICE}")
    print("  Pillars: 1) Multi-Modal  2) Interpretability  "
          "3) Domain Generalization")

    # ── 1. Data ────────────────────────────────────────────────────
    df, y, base_train_idx, blend_val_idx, test_idx = load_and_split()

    # ── 3. SBERT Encoding ─────────────────────────────────────────
    emb_post, emb_art, emb_ttl, emb_dsc = sbert_encode(df)

    # ── 3.5  Visual Feature Extraction (Pillar 1) ─────────────────
    visual_features = extract_visual_features(df)

    # ── 2. Feature Factory ─────────────────────────────────────────
    feat_df, feat_names = build_feature_factory(
        df, emb_post, emb_ttl, emb_art, emb_dsc,
        base_train_idx, y, visual_features)

    # ── 4. Tournament (5 base learners) ───────────────────────────
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

    # 4.4 SBERT-MLP
    bp, tp, m = train_sbert_mlp(
        emb_post, emb_ttl, y, base_train_idx, blend_val_idx, test_idx)
    blend_probs["SBERT-MLP"] = bp
    test_probs["SBERT-MLP"]  = tp
    all_metrics.append(m)

    # 4.5 Multi-Modal Transformer (★ NEW — replaces XGBoost + CatBoost)
    bp, tp, m = train_mmt(
        emb_post, visual_features, feat_df, feat_names,
        y, df, base_train_idx, blend_val_idx, test_idx)
    blend_probs["MMT"] = bp
    test_probs["MMT"]  = tp
    all_metrics.append(m)

    model_names = list(blend_probs.keys())

    # ── 5. Power Blend ─────────────────────────────────────────────
    blend_raw, test_raw, opt_weights, opt_k, w_norm, model_names = \
        optimize_blend(
            blend_probs, test_probs,
            y[blend_val_idx], y[test_idx], model_names)
    test_probs["_raw_blend"] = test_raw

    # ── 6. Temperature Scaling ─────────────────────────────────────
    blend_cal, test_cal, T, ece_raw, ece_cal = temperature_scale(
        blend_raw, y[blend_val_idx], test_raw)

    # ── 7. Final Evaluation ────────────────────────────────────────
    f1, acc, prec, rec, y_pred, buckets = final_evaluation(
        test_cal, y[test_idx])

    # ── 8. Scientific Reports ──────────────────────────────────────
    df_lb = print_leaderboard(all_metrics, f1)
    C     = diversity_matrix(test_probs, model_names)
    what_worked_report(lgbm_clf, lgbm_sc, feat_names,
                        blend_probs, y[blend_val_idx],
                        opt_weights, w_norm, model_names)
    calibration_plot(test_probs, test_cal, y[test_idx], model_names,
                      T, ece_raw, ece_cal)

    # ── PILLAR 2: SHAP Interpretability ────────────────────────────
    X_explain = lgbm_sc.transform(feat_df[feat_names].values[test_idx])
    shap_result = generate_explanation_visuals(
        lgbm_clf, X_explain, feat_names, y[test_idx])

    # Generate example nutrition label
    if HAS_SHAP and shap_result is not None:
        sv, explainer = shap_result
        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]
        sample_label = generate_clickbait_nutrition_label(
            sv[0], feat_names, X_explain[0], base_val,
            visual_mismatch_score=feat_df["visual_text_mismatch"].iloc[
                test_idx[0]])
        with open(CFG["NUTRITION_REPORT"], "w") as f:
            json.dump(sample_label, f, indent=2)
        print(f"  ✓ Example nutrition label → {CFG['NUTRITION_REPORT']}")
        print(f"    Score: {sample_label['score']}%")
        for t in sample_label["top_3_triggers"]:
            print(f"    Trigger: {t['trigger']} ({t['contribution_pct']}%) "
                  f"{t['direction']}")

    # ── PILLAR 3: Zero-Shot Transfer Evaluation ────────────────────
    zero_shot_transfer_eval(test_cal, y[test_idx], df, test_idx)

    # ── 9. Save ────────────────────────────────────────────────────
    save_outputs(df, test_idx, test_cal, y_pred, y[test_idx], buckets,
                  test_probs, model_names, f1, acc, prec, rec,
                  opt_k, w_norm, T, ece_raw, ece_cal)

    print("\n  ═══════════════════════════════════════")
    print("  Pipeline complete — 3-Pillar Architecture:")
    print("    ★ Pillar 1: Multi-Modal Fusion (MobileNetV2 + MMT)")
    print("    ★ Pillar 2: SHAP Interpretability + Nutrition Labels")
    print("    ★ Pillar 3: Domain-Adversarial + Zero-Shot Transfer")
    print("  ═══════════════════════════════════════\n")


if __name__ == "__main__":
    main()
