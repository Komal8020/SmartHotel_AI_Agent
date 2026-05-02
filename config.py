"""
SmartHotel Agent - Configuration & Shared Data Models
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
TRIPADVISOR_KEY: str = os.getenv("TRIPADVISOR_KEY", "")
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Scoring Weights (from paper Section III-F) ────────────────────────────────
W_PRICE    = 0.25   # w1
W_RATING   = 0.30   # w2
W_SENTIMENT= 0.30   # w3
W_AMENITY  = 0.15   # w4

# ── Sentiment Score Formula (from paper Section III-E) ────────────────────────
SENT_POS_WEIGHT = 0.6
SENT_NEU_WEIGHT = 0.2
SENT_NEG_WEIGHT = 0.6   # NOTE: subtracted (penalises negatives more)

# ── BERT Model ────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_TOKENS = 512

# ── Budget filter tolerance ───────────────────────────────────────────────────
BUDGET_TOLERANCE = 0.10   # ±10 %

# ── Top-K recommendations ─────────────────────────────────────────────────────
TOP_K = 5


@dataclass
class UserQuery:
    destination: str
    checkin: str
    checkout: str
    budget_per_night: float          # INR
    amenities: list[str] = field(default_factory=list)
    room_type: str = "standard"
    trip_purpose: str = "leisure"


@dataclass
class HotelRaw:
    """Raw hotel record returned by the search agent."""
    hotel_id: str
    name: str
    location: str
    price_per_night: float
    rating: float
    review_count: int
    amenities: list[str]
    reviews: list[str]
    image_url: str = ""
    source: str = "serpapi"


@dataclass
class SentimentResult:
    positive: float
    neutral: float
    negative: float
    score: float          # composite S formula


@dataclass
class HotelScored:
    hotel: HotelRaw
    sentiment: SentimentResult
    comparison_score: float
    price_score: float
    amenity_score: float
    explanation: str
    rank: int = 0
