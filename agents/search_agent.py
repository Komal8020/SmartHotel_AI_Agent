"""
SmartHotel Agent — Agent 1: Hotel Search Agent
Fetches real-time hotel data from SerpAPI (Google Hotels) and
a bundled TripAdvisor sample dataset.  Filters by budget ±10 %.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Optional

import requests

from config import (
    SERPAPI_KEY,
    BUDGET_TOLERANCE,
    HotelRaw,
    UserQuery,
)

# ── Fallback / demo hotel data so the app runs without a SerpAPI key ──────────
DEMO_HOTELS: list[dict] = [
    {
        "hotel_id": "demo_001",
        "name": "The Grand Residency",
        "location": "{city}",
        "price_per_night": 3200,
        "rating": 4.5,
        "review_count": 1240,
        "amenities": ["wifi", "pool", "spa", "restaurant", "gym"],
        "reviews": [
            "Absolutely stunning property with impeccable service.",
            "The pool area is beautiful and the staff is very helpful.",
            "Food at the restaurant was outstanding, fresh and flavourful.",
            "Room was spacious and very clean. Will definitely come back.",
            "Great location, walking distance to major attractions.",
        ],
        "image_url": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400",
    },
    {
        "hotel_id": "demo_002",
        "name": "Comfort Inn Express",
        "location": "{city}",
        "price_per_night": 1800,
        "rating": 3.9,
        "review_count": 680,
        "amenities": ["wifi", "breakfast", "parking"],
        "reviews": [
            "Decent place for the price, nothing extraordinary.",
            "Breakfast was average but the room was clean.",
            "AC unit was noisy and woke me up multiple times.",
            "Friendly front-desk staff made check-in quick and easy.",
            "Location not ideal — needed a cab for most things.",
        ],
        "image_url": "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=400",
    },
    {
        "hotel_id": "demo_003",
        "name": "Taj Palace Suites",
        "location": "{city}",
        "price_per_night": 6500,
        "rating": 4.8,
        "review_count": 2100,
        "amenities": ["wifi", "pool", "spa", "restaurant", "gym", "bar", "concierge"],
        "reviews": [
            "Luxury at its finest. Every detail is perfect.",
            "The spa treatments are world-class and very relaxing.",
            "Concierge arranged everything flawlessly for our anniversary.",
            "Breakfast spread is phenomenal — over 40 dishes every morning.",
            "The view from our suite was absolutely breathtaking.",
        ],
        "image_url": "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=400",
    },
    {
        "hotel_id": "demo_004",
        "name": "Budget Stay Central",
        "location": "{city}",
        "price_per_night": 900,
        "rating": 3.2,
        "review_count": 310,
        "amenities": ["wifi", "parking"],
        "reviews": [
            "Very basic but gets the job done for business trips.",
            "Rooms are tiny and need renovation.",
            "Staff was not very responsive to complaints.",
            "Cheap price but you get what you pay for.",
            "Hot water supply was inconsistent.",
        ],
        "image_url": "https://images.unsplash.com/photo-1631049307264-da0ec9d70304?w=400",
    },
    {
        "hotel_id": "demo_005",
        "name": "Heritage Boutique Hotel",
        "location": "{city}",
        "price_per_night": 4200,
        "rating": 4.6,
        "review_count": 870,
        "amenities": ["wifi", "restaurant", "bar", "heritage tours", "gym"],
        "reviews": [
            "A unique experience with beautiful colonial architecture.",
            "The heritage walk organised by the hotel was wonderful.",
            "Restaurant serves authentic local cuisine — a must-try.",
            "Staff dressed in traditional attire added to the ambience.",
            "Slightly dated rooms but the character more than compensates.",
        ],
        "image_url": "https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=400",
    },
    {
        "hotel_id": "demo_006",
        "name": "Urban Nest Business Hotel",
        "location": "{city}",
        "price_per_night": 2600,
        "rating": 4.1,
        "review_count": 540,
        "amenities": ["wifi", "gym", "conference room", "restaurant"],
        "reviews": [
            "Great for business travellers, fast wifi and quiet rooms.",
            "Conference facilities are well-equipped and modern.",
            "Gym is small but has all essential equipment.",
            "Could improve the breakfast variety.",
            "Location near the tech park was very convenient.",
        ],
        "image_url": "https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=400",
    },
    {
        "hotel_id": "demo_007",
        "name": "Seaside Escape Resort",
        "location": "{city}",
        "price_per_night": 5100,
        "rating": 4.4,
        "review_count": 1050,
        "amenities": ["wifi", "pool", "beach access", "restaurant", "spa"],
        "reviews": [
            "Waking up to the sound of waves was magical.",
            "Beach is private and pristine — absolutely lovely.",
            "Service was top-notch. Staff remembered our names.",
            "The seafood restaurant on-site is fantastic.",
            "A bit pricey but absolutely worth it for the experience.",
        ],
        "image_url": "https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=400",
    },
    {
        "hotel_id": "demo_008",
        "name": "Green Valley Eco Resort",
        "location": "{city}",
        "price_per_night": 3500,
        "rating": 4.3,
        "review_count": 430,
        "amenities": ["wifi", "pool", "organic restaurant", "yoga", "nature walks"],
        "reviews": [
            "The most eco-friendly stay I have ever had.",
            "Organic meals were delicious and very fresh.",
            "Yoga sessions at sunrise were incredibly peaceful.",
            "Nature walks guided by the staff were informative.",
            "Rooms are simple but comfortable. Perfect detox.",
        ],
        "image_url": "https://images.unsplash.com/photo-1596436889106-be35e843f974?w=400",
    },
]


class HotelSearchAgent:
    """
    Agent 1 — Hotel Search Agent

    Responsibilities
    ----------------
    1. Call SerpAPI (Google Hotels endpoint) with destination + dates.
    2. Fall back to demo data when no key is provided.
    3. Apply the ±10 % budget filter from the paper.
    4. Return a list of HotelRaw objects.
    """

    def __init__(self) -> None:
        self.has_key = bool(SERPAPI_KEY and SERPAPI_KEY != "your_serpapi_key_here")

    # ── Public interface ──────────────────────────────────────────────────────

    def search(self, query: UserQuery) -> list[HotelRaw]:
        if self.has_key:
            hotels = self._search_serpapi(query)
        else:
            hotels = self._demo_hotels(query)

        return self._filter_by_budget(hotels, query.budget_per_night)

    # ── SerpAPI integration ───────────────────────────────────────────────────

    def _search_serpapi(self, query: UserQuery) -> list[HotelRaw]:
        params = {
            "engine": "google_hotels",
            "q": f"hotels in {query.destination}",
            "check_in_date": query.checkin,
            "check_out_date": query.checkout,
            "currency": "INR",
            "gl": "in",
            "hl": "en",
            "api_key": SERPAPI_KEY,
        }
        try:
            resp = requests.get(
                "https://serpapi.com/search", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_serpapi_response(data, query.destination)
        except Exception as exc:
            print(f"[HotelSearchAgent] SerpAPI error: {exc}. Falling back to demo data.")
            return self._demo_hotels(query)

    def _parse_serpapi_response(
        self, data: dict, destination: str
    ) -> list[HotelRaw]:
        hotels: list[HotelRaw] = []
        properties = data.get("properties", [])
        for idx, prop in enumerate(properties[:20]):  # cap at 20
            price_str = prop.get("rate_per_night", {}).get("lowest", "0")
            price = float(re.sub(r"[^\d.]", "", str(price_str)) or 0)
            hotels.append(
                HotelRaw(
                    hotel_id=f"serp_{idx}_{prop.get('name', 'hotel')[:8]}",
                    name=prop.get("name", "Unknown Hotel"),
                    location=destination,
                    price_per_night=price,
                    rating=float(prop.get("overall_rating", 0)),
                    review_count=int(prop.get("reviews", 0)),
                    amenities=[a.lower() for a in prop.get("amenities", [])],
                    reviews=self._extract_reviews(prop),
                    image_url=prop.get("images", [{}])[0].get("thumbnail", ""),
                    source="serpapi",
                )
            )
        return hotels

    @staticmethod
    def _extract_reviews(prop: dict) -> list[str]:
        reviews: list[str] = []
        for snippet in prop.get("review_snippets", []):
            text = snippet.get("snippet", "")
            if text:
                reviews.append(text)
        # pad with generic if none found
        if not reviews:
            reviews = [
                f"Good hotel in a convenient location.",
                f"Comfortable stay with friendly staff.",
            ]
        return reviews

    # ── Demo data (no API key) ────────────────────────────────────────────────

    def _demo_hotels(self, query: UserQuery) -> list[HotelRaw]:
        city = query.destination.split(",")[0].strip().title()
        hotels: list[HotelRaw] = []
        for raw in DEMO_HOTELS:
            # Vary prices slightly per query so results feel dynamic
            price_variance = random.uniform(0.85, 1.15)
            hotels.append(
                HotelRaw(
                    hotel_id=raw["hotel_id"],
                    name=raw["name"],
                    location=city,
                    price_per_night=round(raw["price_per_night"] * price_variance),
                    rating=raw["rating"],
                    review_count=raw["review_count"],
                    amenities=raw["amenities"],
                    reviews=raw["reviews"],
                    image_url=raw["image_url"],
                    source="demo",
                )
            )
        return hotels

    # ── Budget filter (paper: ±10 %) ─────────────────────────────────────────

    @staticmethod
    def _filter_by_budget(
        hotels: list[HotelRaw], budget: float
    ) -> list[HotelRaw]:
        low = budget * (1 - BUDGET_TOLERANCE)
        high = budget * (1 + BUDGET_TOLERANCE)
        filtered = [h for h in hotels if low <= h.price_per_night <= high]
        # If too strict, relax the filter and return all sorted by closeness
        if len(filtered) < 3:
            filtered = sorted(
                hotels, key=lambda h: abs(h.price_per_night - budget)
            )[:8]
        return filtered
