"""
SmartHotel Agent — Agent 5: Recommendation Agent

Selects top-K hotels and generates natural-language explanations
(Section III-G).  Explanation format mirrors the paper example:
  "Ranked #1 with a sentiment score of 0.92, within ₹3200/night,
   meets 4 out of 5 facility criteria."
"""
from __future__ import annotations

from config import TOP_K, HotelScored, UserQuery


class RecommendationAgent:
    """
    Agent 5 — Recommendation Agent

    Input  : list[HotelScored] (already sorted by comparison score)
    Output : list[HotelScored] — top-K with explanations attached
    """

    def recommend(
        self, scored: list[HotelScored], query: UserQuery
    ) -> list[HotelScored]:
        top = scored[:TOP_K]
        for rank, hotel_scored in enumerate(top, start=1):
            hotel_scored.rank = rank
            hotel_scored.explanation = self._explain(hotel_scored, query, rank)
        return top

    # ── Natural-language explanation ──────────────────────────────────────────

    @staticmethod
    def _explain(hs: HotelScored, query: UserQuery, rank: int) -> str:
        h = hs.hotel
        amenity_total = len(query.amenities) if query.amenities else 5
        amenity_matched = round(hs.amenity_score * amenity_total)

        # Sentiment narrative
        s = hs.sentiment
        if s.positive >= 0.65:
            sent_desc = "highly positive guest sentiment"
        elif s.positive >= 0.45:
            sent_desc = "mostly positive guest sentiment"
        else:
            sent_desc = "mixed guest sentiment"

        # Price narrative
        diff = h.price_per_night - query.budget_per_night
        if abs(diff) < 50:
            price_desc = f"exactly within your ₹{query.budget_per_night:,.0f}/night budget"
        elif diff < 0:
            price_desc = f"₹{abs(diff):,.0f} below your budget"
        else:
            price_desc = f"₹{abs(diff):,.0f} above your budget"

        explanation = (
            f"Ranked #{rank} with a sentiment score of {s.score:.2f}, "
            f"{price_desc}, and meets {amenity_matched} out of "
            f"{amenity_total} requested amenities. "
            f"The hotel shows {sent_desc} "
            f"({s.positive * 100:.0f}% positive reviews). "
            f"Overall composite score: {hs.comparison_score:.3f}."
        )
        return explanation
