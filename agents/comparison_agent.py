"""
SmartHotel Agent — Agent 4: Comparison Agent

Implements the composite scoring function C(h) from Section III-F:

    C(h) = w1 * Price + w2 * Rating + w3 * S + w4 * Amenity

where
    w1 = 0.25  (price score — inverse, normalised within budget band)
    w2 = 0.30  (platform rating normalised to [0, 1])
    w3 = 0.30  (sentiment score S from Agent 3)
    w4 = 0.15  (amenity match fraction from Agent 2)
"""
from __future__ import annotations

from config import (
    W_AMENITY,
    W_PRICE,
    W_RATING,
    W_SENTIMENT,
    HotelRaw,
    HotelScored,
    SentimentResult,
    UserQuery,
)
from agents.data_processing_agent import ProcessedHotel


class ComparisonAgent:
    """
    Agent 4 — Comparison Agent

    Input  : list[ProcessedHotel], sentiment results, UserQuery
    Output : list[HotelScored] sorted by C(h) descending
    """

    def compare(
        self,
        processed: list[ProcessedHotel],
        sentiments: dict[str, SentimentResult],
        query: UserQuery,
    ) -> list[HotelScored]:
        # Collect raw prices to normalise
        prices = [ph.hotel.price_per_night for ph in processed]
        min_price = min(prices) if prices else 1
        max_price = max(prices) if prices else 1

        scored: list[HotelScored] = []
        for ph in processed:
            hotel = ph.hotel
            sentiment = sentiments.get(
                hotel.hotel_id,
                SentimentResult(positive=0.5, neutral=0.3, negative=0.2, score=0.22),
            )

            price_score = self._price_score(hotel.price_per_night, min_price, max_price)
            rating_score = hotel.rating / 5.0   # normalise 5-star to [0,1]
            amenity_score = ph.amenity_match_score
            sent_score = max(0.0, min(1.0, (sentiment.score + 0.6) / 1.2))  # map to [0,1]

            composite = (
                W_PRICE     * price_score
                + W_RATING  * rating_score
                + W_SENTIMENT * sent_score
                + W_AMENITY * amenity_score
            )

            scored.append(
                HotelScored(
                    hotel=hotel,
                    sentiment=sentiment,
                    comparison_score=round(composite, 4),
                    price_score=round(price_score, 4),
                    amenity_score=round(amenity_score, 4),
                    explanation="",   # filled by RecommendationAgent
                )
            )

        # Sort descending by composite score
        scored.sort(key=lambda h: h.comparison_score, reverse=True)
        return scored

    # ── Price score (inverse: cheaper → higher score within band) ─────────────

    @staticmethod
    def _price_score(price: float, min_p: float, max_p: float) -> float:
        if max_p == min_p:
            return 1.0
        # Normalise then invert: low price → high score
        normalised = (price - min_p) / (max_p - min_p)
        return round(1.0 - normalised, 4)
