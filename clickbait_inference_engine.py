"""
clickbait_inference_engine.py
------------------------------
Production-ready, modular inference engine for the fine-tuned DistilBERT
3-class clickbait classifier.

Schema
------
  Class 0: Not Clickbait  (truthMean < 0.30)
  Class 1: Ambiguous      (truthMean 0.30 - 0.70)
  Class 2: Clickbait      (truthMean >= 0.70)

The model was trained with input:
    [postText] [SEP] [targetTitle]

Public API
----------
  InputProcessor          - normalise any upstream data shape
  ClickbaitInferenceEngine - singleton that loads weights once
      .predict_single(record)  -> structured dict
      .predict_batch(records)  -> list[dict]
  web_extension_mock(url, headline) -> safety dict

Usage
-----
  engine = ClickbaitInferenceEngine(weights_path="clickbait_distilbert_v5.pt")
  result = engine.predict_single({"postText": "You won't believe this!"})
  print(result)
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ---------------------------------------------------------------------------
# Constants  (mirror the training config so nothing diverges)
# ---------------------------------------------------------------------------
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 96
NUM_LABELS   = 3
FALLBACK_CTX = "No Context Available"

_LABEL_INT_TO_STR = {0: "Not Clickbait", 1: "Ambiguous", 2: "Clickbait"}
_LABEL_SHORT      = {0: "Not",           1: "Ambiguous", 2: "Clickbait"}

# Web-extension safety thresholds (probability of class 2 / Clickbait)
_SAFETY_GREEN  = 0.30   # P(clickbait) below this  -> safe
_SAFETY_YELLOW = 0.65   # P(clickbait) above this  -> red; between thresholds -> yellow


# ===========================================================================
# 1. INPUT PROCESSOR
# ===========================================================================

class InputProcessor:
    """
    Normalise heterogeneous upstream records into (post_text, context_text)
    pairs that the tokeniser can consume.

    Accepted input shapes
    ---------------------
    • str                        - treated as postText; context falls back
    • dict with 'postText'       - optional 'targetTitle' used as context
    • dict with 'headline'       - alias for postText (web-scraper convention)
    • dict with 'text'           - generic alias for postText
    • list[str]                  - joined into a single postText string

    Fallback context logic
    ----------------------
    If no context (targetTitle / url / etc.) is available, the first
    MAX_FALLBACK_TOKENS whitespace-separated tokens of postText are used as a
    pseudo-context.  This mirrors the training format closely enough that the
    model behaves stably rather than seeing an empty second segment.
    """

    MAX_FALLBACK_TOKENS: int = 50

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self,
        record: Union[str, Dict[str, Any], List[str]],
    ) -> tuple[str, str]:
        """
        Return (post_text, context_text) ready for the tokeniser.

        Parameters
        ----------
        record : str | dict | list[str]
            Raw input from any upstream source.

        Returns
        -------
        (post_text, context_text) : tuple[str, str]
            Both strings are guaranteed to be non-empty.
        """
        post_text, context_text = self._extract(record)
        post_text    = self._clean(post_text)
        context_text = self._clean(context_text)

        if not post_text:
            post_text = FALLBACK_CTX

        if not context_text:
            context_text = self._pseudo_context(post_text)

        return post_text, context_text

    def process_batch(
        self,
        records: List[Union[str, Dict[str, Any], List[str]]],
    ) -> tuple[List[str], List[str]]:
        """Vectorised version of :meth:`process`."""
        pairs = [self.process(r) for r in records]
        posts, contexts = zip(*pairs)
        return list(posts), list(contexts)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract(
        self,
        record: Union[str, Dict[str, Any], List[str]],
    ) -> tuple[str, str]:
        """Pull raw (post_text, context_text) from the input without cleaning."""
        if isinstance(record, str):
            return record, ""

        if isinstance(record, list):
            return " ".join(str(t) for t in record), ""

        if isinstance(record, dict):
            # Resolve post_text from common field aliases
            post_text = (
                record.get("postText")
                or record.get("headline")
                or record.get("text")
                or record.get("title")
                or ""
            )
            # Resolve context from common field aliases
            context_text = (
                record.get("targetTitle")
                or record.get("context")
                or record.get("article_title")
                or record.get("source_title")
                or ""
            )
            # Flatten list-encoded fields (training CSV stores lists as strings)
            if isinstance(post_text, list):
                post_text = " ".join(str(t) for t in post_text)
            if isinstance(context_text, list):
                context_text = " ".join(str(t) for t in context_text)

            return str(post_text), str(context_text)

        return str(record), ""

    @staticmethod
    def _clean(text: str) -> str:
        return text.strip()

    def _pseudo_context(self, post_text: str) -> str:
        """First N tokens of postText as a stand-in context segment."""
        tokens = post_text.split()
        if len(tokens) <= self.MAX_FALLBACK_TOKENS:
            return post_text
        return " ".join(tokens[: self.MAX_FALLBACK_TOKENS])


# ===========================================================================
# 2. SINGLETON INFERENCE ENGINE
# ===========================================================================

class ClickbaitInferenceEngine:
    """
    Thread-safe singleton that loads the DistilBERT weights exactly once
    and exposes predict_single / predict_batch.

    Parameters (passed to the *first* call only)
    --------------------------------------------
    weights_path : str
        Path to a ``torch.save(model.state_dict(), ...)`` checkpoint.
        Pass ``None`` to run with freshly initialised (random) weights —
        useful for unit-testing the pipeline without a real checkpoint.
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    batch_size : int
        Internal mini-batch size for ``predict_batch``.

    Example
    -------
    >>> engine = ClickbaitInferenceEngine("clickbait_distilbert_v5.pt")
    >>> engine.predict_single({"postText": "Doctors HATE this one trick!"})
    {
        "final_prediction": "Clickbait",
        "confidence_score": 0.94,
        "class_probabilities": {"Not": 0.02, "Ambiguous": 0.04, "Clickbait": 0.94},
        "metadata": {"tokens_processed": 11, "runtime_ms": 8}
    }
    """

    _instance: Optional["ClickbaitInferenceEngine"] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(
        cls,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> "ClickbaitInferenceEngine":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialised = False
                cls._instance = instance
            return cls._instance

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        # Guard: only initialise once even if __init__ is called again
        if self._initialised:
            return

        self.batch_size  = batch_size
        self.device      = self._resolve_device(device)
        self._processor  = InputProcessor()
        self._tokeniser  = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self._model      = self._load_model(weights_path)
        self._initialised = True

        print(
            f"[ClickbaitInferenceEngine] Ready  |  device={self.device}  |  "
            f"weights={'<random>' if weights_path is None else weights_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(
        self,
        record: Union[str, Dict[str, Any], List[str]],
    ) -> Dict[str, Any]:
        """
        Classify a single headline/post record.

        Parameters
        ----------
        record : str | dict | list[str]
            Any shape accepted by :class:`InputProcessor`.

        Returns
        -------
        dict
            Structured prediction with keys:
            ``final_prediction``, ``confidence_score``,
            ``class_probabilities``, ``metadata``.
        """
        results = self.predict_batch([record])
        return results[0]

    def predict_batch(
        self,
        records: List[Union[str, Dict[str, Any], List[str]]],
    ) -> List[Dict[str, Any]]:
        """
        Classify a list of records efficiently in mini-batches.

        Parameters
        ----------
        records : list
            Each element can be any shape accepted by :class:`InputProcessor`.

        Returns
        -------
        list[dict]
            One structured-prediction dict per input record.
        """
        if not records:
            return []

        t_start = time.perf_counter()

        post_texts, context_texts = self._processor.process_batch(records)
        all_probs = self._run_inference(post_texts, context_texts)

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        # Build per-record token counts
        token_counts = self._count_tokens(post_texts, context_texts)

        results = []
        per_record_ms = round(total_ms / len(records), 1)

        for i, probs in enumerate(all_probs):
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx])

            results.append({
                "final_prediction": _LABEL_INT_TO_STR[pred_idx],
                "confidence_score": round(confidence, 4),
                "class_probabilities": {
                    _LABEL_SHORT[j]: round(float(probs[j]), 4)
                    for j in range(NUM_LABELS)
                },
                "metadata": {
                    "tokens_processed": token_counts[i],
                    "runtime_ms": per_record_ms,
                },
            })

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(
        self, weights_path: Optional[str]
    ) -> nn.Module:
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            # Accept both bare state_dict and {"model_state_dict": ...} wrappers
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            print(f"[ClickbaitInferenceEngine] Loaded weights from {weights_path}")

        model.to(self.device)
        model.eval()
        return model

    def _run_inference(
        self,
        post_texts: List[str],
        context_texts: List[str],
    ) -> List[torch.Tensor]:
        """
        Tokenise and forward-pass in mini-batches.

        Returns
        -------
        list[Tensor]   shape (NUM_LABELS,) per record, CPU, softmax probabilities
        """
        all_probs: List[torch.Tensor] = []

        for start in range(0, len(post_texts), self.batch_size):
            batch_posts    = post_texts[start : start + self.batch_size]
            batch_contexts = context_texts[start : start + self.batch_size]

            # Mimic training: text_a=[postText], text_b=[targetTitle/context]
            # The tokeniser inserts [CLS] post [SEP] context [SEP] automatically.
            encoding = self._tokeniser(
                batch_posts,
                batch_contexts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits                              # (batch, 3)
                probs = torch.softmax(logits, dim=-1) # (batch, 3)

            all_probs.extend(probs.cpu().unbind(dim=0))

        return all_probs

    def _count_tokens(
        self,
        post_texts: List[str],
        context_texts: List[str],
    ) -> List[int]:
        """Return the number of non-padding tokens for each pair."""
        encoding = self._tokeniser(
            post_texts,
            context_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # attention_mask is 1 for real tokens, 0 for padding
        return encoding["attention_mask"].sum(dim=1).tolist()


# ===========================================================================
# 3. WEB EXTENSION MOCK INTERFACE
# ===========================================================================

def _safety_rating(clickbait_prob: float) -> str:
    """Map P(clickbait) to a traffic-light safety label."""
    if clickbait_prob < _SAFETY_GREEN:
        return "Green"
    if clickbait_prob < _SAFETY_YELLOW:
        return "Yellow"
    return "Red"


def web_extension_mock(
    url: str,
    headline: str,
    engine: Optional[ClickbaitInferenceEngine] = None,
) -> Dict[str, Any]:
    """
    Simulate how a browser extension / web scraper would consume the engine.

    The scraper extracts a ``headline`` from a page at ``url`` and passes it
    here.  Because there is no article body yet, only the headline is available,
    so the engine's InputProcessor falls back to pseudo-context automatically.

    Parameters
    ----------
    url : str
        Source URL (used as metadata; not fetched).
    headline : str
        The page title or article headline scraped by the extension.
    engine : ClickbaitInferenceEngine | None
        Pass an already-initialised engine to avoid repeated loading.
        If ``None``, the singleton is retrieved (or created with random weights).

    Returns
    -------
    dict
        {
            "url": ...,
            "headline": ...,
            "safety_rating": "Green" | "Yellow" | "Red",
            "prediction": <full engine output dict>,
        }

    Example
    -------
    >>> result = web_extension_mock(
    ...     url="https://example.com/article",
    ...     headline="You won't believe what happened next!",
    ... )
    >>> print(result["safety_rating"])   # "Red"
    """
    if engine is None:
        engine = ClickbaitInferenceEngine()  # returns singleton

    # Only the headline is known; targetTitle is absent — InputProcessor handles it
    record = {"headline": headline}
    prediction = engine.predict_single(record)

    cb_prob = prediction["class_probabilities"]["Clickbait"]
    rating  = _safety_rating(cb_prob)

    return {
        "url": url,
        "headline": headline,
        "safety_rating": rating,
        "prediction": prediction,
    }


# ===========================================================================
# DEMO  (run with:  python clickbait_inference_engine.py)
# ===========================================================================

if __name__ == "__main__":
    import json

    print("=" * 68)
    print("ClickbaitInferenceEngine  —  Demo")
    print("=" * 68)

    # -------------------------------------------------------------------
    # Initialise the singleton.
    # Pass weights_path="clickbait_distilbert_v5.pt" once trained.
    # -------------------------------------------------------------------
    engine = ClickbaitInferenceEngine(weights_path=None)  # random weights for demo

    # -------------------------------------------------------------------
    # A.  predict_single — various input shapes
    # -------------------------------------------------------------------
    print("\n--- predict_single: string input ---")
    result = engine.predict_single("Scientists reveal the SHOCKING truth about coffee!")
    print(json.dumps(result, indent=2))

    print("\n--- predict_single: dict with both fields ---")
    result = engine.predict_single({
        "postText":    "You won't believe what celebrities eat for breakfast",
        "targetTitle": "Celebrity Breakfast Habits",
    })
    print(json.dumps(result, indent=2))

    print("\n--- predict_single: dict with only postText (fallback active) ---")
    result = engine.predict_single({
        "postText": "New study shows walking 10 minutes a day improves heart health",
    })
    print(json.dumps(result, indent=2))

    # -------------------------------------------------------------------
    # B.  predict_batch — heterogeneous records in one call
    # -------------------------------------------------------------------
    print("\n--- predict_batch: mixed input shapes ---")
    batch = [
        "Doctors HATE him! Local man discovers ancient remedy",
        {"postText": "Senate passes bipartisan infrastructure bill",
         "targetTitle": "Infrastructure Investment and Jobs Act"},
        {"headline": "Is your smartphone SECRETLY recording you?"},
        ["This", "is", "a", "tokenised", "post", "text"],
    ]
    results = engine.predict_batch(batch)
    for i, r in enumerate(results):
        print(f"\n  Record {i+1}: {r['final_prediction']}  "
              f"(conf={r['confidence_score']:.2f}, "
              f"tokens={r['metadata']['tokens_processed']}, "
              f"ms={r['metadata']['runtime_ms']})")
        print(f"    Probs: {r['class_probabilities']}")

    # -------------------------------------------------------------------
    # C.  web_extension_mock — simulated browser extension call
    # -------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Web Extension Mock Interface")
    print("=" * 68)

    pages = [
        ("https://news.example.com/politics/senate-vote",
         "Senate votes 72-28 to pass infrastructure bill"),
        ("https://viral.example.com/health/secret",
         "Doctors are FURIOUS about this one weird trick"),
        ("https://tech.example.com/ai-update",
         "OpenAI releases incremental update to GPT model"),
    ]

    for url, headline in pages:
        ext_result = web_extension_mock(url, headline, engine=engine)
        rating = ext_result["safety_rating"]
        badge  = {"Green": "✓ SAFE", "Yellow": "⚠ CAUTION", "Red": "✗ CLICKBAIT"}[rating]
        print(f"\n  [{badge}]  {rating}")
        print(f"  URL:      {url}")
        print(f"  Headline: {headline}")
        print(f"  P(CB)={ext_result['prediction']['class_probabilities']['Clickbait']:.3f}  "
              f"Prediction: {ext_result['prediction']['final_prediction']}")

    print("\n" + "=" * 68)
    print("Demo complete.")
    print("=" * 68)
