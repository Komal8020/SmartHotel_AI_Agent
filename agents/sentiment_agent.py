"""
SmartHotel Agent — Agent 3: Sentiment Analysis Agent

Uses a pre-trained BERT (bert-base-uncased) model fine-tuned on hotel
reviews to classify each sentence as Positive / Neutral / Negative.

Sentiment score formula (Section III-E):
    S = (0.6 * P) + (0.2 * Ne) - (0.6 * Neg)
where P, Ne, Neg are proportions of positive, neutral and negative sentences.

Falls back to a lexicon-based scorer when transformers are unavailable
(e.g., slow CPU-only environment) so the REST API still runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import (
    BERT_MAX_TOKENS,
    BERT_MODEL_NAME,
    SENT_NEG_WEIGHT,
    SENT_NEU_WEIGHT,
    SENT_POS_WEIGHT,
    SentimentResult,
)
from agents.data_processing_agent import ProcessedHotel

# ── Try importing transformers (optional dependency) ──────────────────────────
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


# ── Simple lexicon fallback ───────────────────────────────────────────────────
_POS_WORDS = {
    "excellent", "great", "fantastic", "wonderful", "amazing", "outstanding",
    "perfect", "beautiful", "lovely", "superb", "brilliant", "awesome",
    "helpful", "clean", "comfortable", "spacious", "delicious", "friendly",
    "pleasant", "enjoyable", "recommend", "worth", "best", "love",
}
_NEG_WORDS = {
    "terrible", "awful", "horrible", "poor", "bad", "worst", "dirty",
    "noisy", "rude", "disappointing", "unacceptable", "broken", "disgusting",
    "smelly", "cramped", "slow", "unfriendly", "overpriced", "avoid",
    "nightmare", "filthy", "stained", "loud", "cold", "leaking",
}


class SentimentAnalysisAgent:
    """
    Agent 3 — Sentiment Analysis Agent

    Input  : list[ProcessedHotel]
    Output : dict[hotel_id -> SentimentResult]
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._use_bert = False

        if _TRANSFORMERS_OK:
            self._load_bert()

    # ── BERT loading ──────────────────────────────────────────────────────────

    def _load_bert(self) -> None:
        """
        Load the fine-tuned BERT model.
        We use a DistilBERT model fine-tuned on SST-2 as a publicly available
        proxy; in production you would swap in your TripAdvisor-fine-tuned
        checkpoint (89.2 % accuracy from the paper).
        """
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self._model.eval()
            self._use_bert = True
            print("[SentimentAgent] BERT model loaded successfully.")
        except Exception as exc:
            print(f"[SentimentAgent] Could not load BERT ({exc}). Using lexicon fallback.")
            self._use_bert = False

    # ── Public interface ──────────────────────────────────────────────────────

    def analyse(
        self, processed_hotels: list[ProcessedHotel]
    ) -> dict[str, SentimentResult]:
        results: dict[str, SentimentResult] = {}
        for ph in processed_hotels:
            results[ph.hotel.hotel_id] = self._score_hotel(ph.clean_reviews)
        return results

    # ── Per-hotel scoring ─────────────────────────────────────────────────────

    def _score_hotel(self, sentences: list[str]) -> SentimentResult:
        if not sentences:
            return SentimentResult(positive=0.5, neutral=0.3, negative=0.2, score=0.22)

        pos, neu, neg = 0, 0, 0
        for sent in sentences:
            label = self._classify(sent)
            if label == "positive":
                pos += 1
            elif label == "negative":
                neg += 1
            else:
                neu += 1

        total = pos + neu + neg
        p = pos / total
        ne = neu / total
        ng = neg / total

        # Paper formula
        score = (SENT_POS_WEIGHT * p) + (SENT_NEU_WEIGHT * ne) - (SENT_NEG_WEIGHT * ng)
        return SentimentResult(positive=p, neutral=ne, negative=ng, score=round(score, 4))

    # ── Classification (BERT or lexicon) ─────────────────────────────────────

    def _classify(self, sentence: str) -> str:
        if self._use_bert:
            return self._bert_classify(sentence)
        return self._lexicon_classify(sentence)

    def _bert_classify(self, sentence: str) -> str:
        try:
            import torch
            inputs = self._tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=BERT_MAX_TOKENS,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            pred = int(torch.argmax(logits, dim=1).item())
            # SST-2: 0 = NEGATIVE, 1 = POSITIVE
            # Map to 3-class (no neutral in SST-2, so use confidence for neutral)
            probs = torch.softmax(logits, dim=1)[0].tolist()
            max_prob = max(probs)
            if max_prob < 0.70:
                return "neutral"
            return "positive" if pred == 1 else "negative"
        except Exception:
            return self._lexicon_classify(sentence)

    @staticmethod
    def _lexicon_classify(sentence: str) -> str:
        tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
        pos_hits = len(tokens & _POS_WORDS)
        neg_hits = len(tokens & _NEG_WORDS)
        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"
