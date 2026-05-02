"""
SmartHotel Agent — FastAPI Backend  (backend/main.py)

Endpoints
─────────
POST /recommend          → run full 5-agent pipeline
GET  /health             → liveness check
GET  /agents/status      → which agents are active
"""
from __future__ import annotations

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from config import UserQuery
from orchestrator import SmartHotelOrchestrator

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartHotel Agent API",
    description=(
        "Multi-agent AI system for intelligent hotel discovery "
        "and recommendation (as described in CIE3 Research Paper)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate orchestrator once (loads BERT model at startup)
orchestrator = SmartHotelOrchestrator()

# ── Request / Response schemas ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    destination: str = Field(..., example="Bangalore")
    checkin: str = Field(..., example="2025-08-01")
    checkout: str = Field(..., example="2025-08-03")
    budget_per_night: float = Field(..., gt=0, example=3500)
    amenities: list[str] = Field(default=[], example=["wifi", "pool", "gym"])
    room_type: str = Field(default="standard", example="deluxe")
    trip_purpose: str = Field(default="leisure", example="business")


class SentimentInfo(BaseModel):
    positive: float
    neutral: float
    negative: float
    score: float


class HotelRecommendation(BaseModel):
    rank: int
    hotel_id: str
    name: str
    location: str
    price_per_night: float
    rating: float
    review_count: int
    amenities: list[str]
    image_url: str
    sentiment: SentimentInfo
    comparison_score: float
    price_score: float
    amenity_score: float
    explanation: str


class RecommendResponse(BaseModel):
    city: str
    query_budget: float
    elapsed_seconds: float
    total_recommendations: int
    recommendations: list[HotelRecommendation]
    pipeline_note: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartHotel Agent API"}


@app.get("/agents/status")
def agents_status():
    bert_active = orchestrator.sentiment_agent._use_bert
    serpapi_active = orchestrator.search_agent.has_key
    return {
        "agents": {
            "1_hotel_search": {
                "status": "active",
                "mode": "SerpAPI (live)" if serpapi_active else "Demo data (no key)",
            },
            "2_data_processing": {"status": "active", "mode": "NLTK + TF-IDF"},
            "3_sentiment_analysis": {
                "status": "active",
                "mode": "BERT (DistilBERT)" if bert_active else "Lexicon fallback",
            },
            "4_comparison": {"status": "active", "mode": "Weighted C(h) scoring"},
            "5_recommendation": {"status": "active", "mode": "Top-K + NL explanation"},
        }
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    query = UserQuery(
        destination=req.destination,
        checkin=req.checkin,
        checkout=req.checkout,
        budget_per_night=req.budget_per_night,
        amenities=req.amenities,
        room_type=req.room_type,
        trip_purpose=req.trip_purpose,
    )

    try:
        result = orchestrator.run(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not result.recommendations:
        raise HTTPException(
            status_code=404,
            detail="No hotels found matching your criteria.",
        )

    recs = []
    for hs in result.recommendations:
        recs.append(
            HotelRecommendation(
                rank=hs.rank,
                hotel_id=hs.hotel.hotel_id,
                name=hs.hotel.name,
                location=hs.hotel.location,
                price_per_night=hs.hotel.price_per_night,
                rating=hs.hotel.rating,
                review_count=hs.hotel.review_count,
                amenities=hs.hotel.amenities,
                image_url=hs.hotel.image_url,
                sentiment=SentimentInfo(**{
                    "positive": hs.sentiment.positive,
                    "neutral": hs.sentiment.neutral,
                    "negative": hs.sentiment.negative,
                    "score": hs.sentiment.score,
                }),
                comparison_score=hs.comparison_score,
                price_score=hs.price_score,
                amenity_score=hs.amenity_score,
                explanation=hs.explanation,
            )
        )

    serpapi_note = (
        "Live SerpAPI data" if orchestrator.search_agent.has_key
        else "Demo data — add SERPAPI_KEY to .env for live results"
    )
    bert_note = (
        "BERT sentiment analysis"
        if orchestrator.sentiment_agent._use_bert
        else "Lexicon-based sentiment (install transformers for BERT)"
    )

    return RecommendResponse(
        city=result.city,
        query_budget=req.budget_per_night,
        elapsed_seconds=result.elapsed_seconds,
        total_recommendations=len(recs),
        recommendations=recs,
        pipeline_note=f"{serpapi_note} | {bert_note}",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
