"""
SmartHotel Agent — Agent 2: Data Processing & Analysis Agent

Implements the four pre-processing steps from Section III-D:
  i.   Lower-case + unicode normalisation
  ii.  Strip special / non-alphanumeric symbols & stop-words (NLTK)
  iii. Sentence splitting
  iv.  TF-IDF amenity / feature extraction
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from config import HotelRaw, UserQuery

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

try:
    STOP_WORDS: set[str] = set(stopwords.words("english"))
except Exception:
    # Minimal fallback stopword list if NLTK data unavailable
    STOP_WORDS = {
        "i","me","my","we","our","you","he","she","it","they","them","their",
        "what","which","who","this","that","these","those","am","is","are","was",
        "were","be","been","being","have","has","had","do","does","did","will",
        "would","could","should","may","might","shall","can","a","an","the",
        "and","but","or","nor","for","so","yet","in","on","at","to","for",
        "of","with","by","from","up","about","into","through","during","before",
        "after","above","below","between","out","off","over","under","again",
        "then","once","here","there","when","where","why","how","all","both",
        "each","few","more","most","other","some","such","no","not","only","own",
        "same","than","too","very","s","t","just","don","now","also","as","if",
    }


@dataclass
class ProcessedHotel:
    hotel: HotelRaw
    clean_reviews: list[str]       # sentence-level, cleaned
    feature_vector: dict[str, float]  # TF-IDF scores for amenity terms
    amenity_match_score: float     # fraction of requested amenities present


class DataProcessingAgent:
    """
    Agent 2 — Data Processing & Analysis Agent

    Input  : list[HotelRaw]  +  UserQuery
    Output : list[ProcessedHotel]
    """

    AMENITY_KEYWORDS = [
        "wifi", "pool", "spa", "gym", "restaurant", "bar", "parking",
        "breakfast", "beach", "conference", "yoga", "concierge",
        "pet-friendly", "rooftop", "jacuzzi", "shuttle", "laundry",
    ]

    def process(
        self, hotels: list[HotelRaw], query: UserQuery
    ) -> list[ProcessedHotel]:
        processed: list[ProcessedHotel] = []
        all_review_docs = [" ".join(h.reviews) for h in hotels]

        # Fit a single TF-IDF vectoriser over all review corpora
        vectoriser = TfidfVectorizer(
            vocabulary=self.AMENITY_KEYWORDS,
            stop_words="english",
        )
        try:
            tfidf_matrix = vectoriser.fit_transform(all_review_docs)
        except Exception:
            tfidf_matrix = None

        for idx, hotel in enumerate(hotels):
            clean_reviews = self._clean_reviews(hotel.reviews)
            feature_vector = {}
            if tfidf_matrix is not None:
                row = tfidf_matrix[idx].toarray()[0]
                feature_vector = {
                    term: float(score)
                    for term, score in zip(self.AMENITY_KEYWORDS, row)
                }
            amenity_score = self._compute_amenity_match(
                hotel.amenities, query.amenities
            )
            processed.append(
                ProcessedHotel(
                    hotel=hotel,
                    clean_reviews=clean_reviews,
                    feature_vector=feature_vector,
                    amenity_match_score=amenity_score,
                )
            )
        return processed

    # ── Text cleaning pipeline ────────────────────────────────────────────────

    def _clean_reviews(self, reviews: list[str]) -> list[str]:
        sentences: list[str] = []
        for review in reviews:
            # i. Lower-case + unicode normalise
            text = unicodedata.normalize("NFKD", review.lower())
            # ii. Strip HTML, special chars
            text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
            text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)  # non-alphanumeric
            text = re.sub(r"\s+", " ", text).strip()
            # iii. Sentence split
            for sent in self._split_sentences(text):
                clean = self._remove_stopwords(sent)
                if len(clean.split()) >= 3:
                    sentences.append(clean)
        return sentences

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        try:
            return sent_tokenize(text)
        except Exception:
            return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

    @staticmethod
    def _remove_stopwords(sentence: str) -> str:
        tokens = sentence.split()
        return " ".join(t for t in tokens if t not in STOP_WORDS)

    # ── Amenity match score ───────────────────────────────────────────────────

    @staticmethod
    def _compute_amenity_match(
        hotel_amenities: list[str], requested: list[str]
    ) -> float:
        if not requested:
            return 1.0
        hotel_lower = {a.lower() for a in hotel_amenities}
        matches = sum(
            1 for req in requested if req.lower() in hotel_lower
        )
        return matches / len(requested)
