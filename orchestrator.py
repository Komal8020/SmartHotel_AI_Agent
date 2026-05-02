"""
SmartHotel Agent — LangChain Orchestrator

Coordinates the five-agent pipeline described in Section III:

  1. HotelSearchAgent      → raw hotel data
  2. DataProcessingAgent   → cleaned reviews + feature vectors
  3. SentimentAnalysisAgent→ sentiment scores
  4. ComparisonAgent       → composite C(h) scores
  5. RecommendationAgent   → top-K + explanations

Uses LangChain's Sequential chain pattern so each agent's output
is the next agent's input, maintaining the data-flow described
in Figure 2 of the paper.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from config import HotelScored, UserQuery
from agents.search_agent import HotelSearchAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.comparison_agent import ComparisonAgent
from agents.recommendation_agent import RecommendationAgent


@dataclass
class PipelineResult:
    recommendations: list[HotelScored]
    elapsed_seconds: float
    city: str
    query: UserQuery


class SmartHotelOrchestrator:
    """
    LangChain-style orchestrator that runs the five agents in sequence.
    Each agent runs independently and can be swapped/scaled individually
    (Section V-E of the paper).
    """

    def __init__(self) -> None:
        self.search_agent    = HotelSearchAgent()
        self.data_agent      = DataProcessingAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.comparison_agent = ComparisonAgent()
        self.rec_agent       = RecommendationAgent()

    def run(self, query: UserQuery) -> PipelineResult:
        t0 = time.perf_counter()

        # ── Agent 1: Hotel Search ─────────────────────────────────────────────
        raw_hotels = self.search_agent.search(query)
        if not raw_hotels:
            return PipelineResult(
                recommendations=[],
                elapsed_seconds=0.0,
                city=query.destination,
                query=query,
            )

        # ── Agent 2: Data Processing ──────────────────────────────────────────
        processed = self.data_agent.process(raw_hotels, query)

        # ── Agent 3: Sentiment Analysis ───────────────────────────────────────
        sentiments = self.sentiment_agent.analyse(processed)

        # ── Agent 4: Comparison ───────────────────────────────────────────────
        scored = self.comparison_agent.compare(processed, sentiments, query)

        # ── Agent 5: Recommendation ───────────────────────────────────────────
        top_k = self.rec_agent.recommend(scored, query)

        elapsed = round(time.perf_counter() - t0, 3)
        return PipelineResult(
            recommendations=top_k,
            elapsed_seconds=elapsed,
            city=query.destination,
            query=query,
        )
