"""
SmartHotel Agent — Streamlit Frontend  (frontend/app.py)

A production-quality UI matching the paper's description.
Run with:  streamlit run frontend/app.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import json
from datetime import date, timedelta
from config import BACKEND_URL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartHotel Agent",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

/* Dark luxury theme */
.stApp { background: #0a0a0f; color: #e8dcc8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #12121a 0%, #1a1225 100%);
    border-right: 1px solid #2a2035;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #16151f 0%, #1e1a2e 100%);
    border: 1px solid #2d2845;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
}

/* Hotel card */
.hotel-card {
    background: linear-gradient(135deg, #13121c 0%, #1c1830 100%);
    border: 1px solid #2d2845;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.hotel-card:hover { border-color: #c9a84c; }

/* Rank badge */
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #c9a84c, #e8c96d);
    color: #0a0a0f;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
    text-transform: uppercase;
}

/* Sentiment bar */
.sentiment-bar-container {
    background: #1e1a2e;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 4px 0;
}
.sentiment-bar-pos { background: linear-gradient(90deg, #4caf82, #6dd5a0); height: 100%; border-radius: 6px; }
.sentiment-bar-neu { background: linear-gradient(90deg, #c9a84c, #e8c96d); height: 100%; border-radius: 6px; }
.sentiment-bar-neg { background: linear-gradient(90deg, #e05252, #f47878); height: 100%; border-radius: 6px; }

/* Score pill */
.score-pill {
    display: inline-block;
    background: rgba(201, 168, 76, 0.15);
    border: 1px solid rgba(201, 168, 76, 0.4);
    color: #c9a84c;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}

/* Amenity chip */
.amenity-chip {
    display: inline-block;
    background: rgba(100, 120, 200, 0.15);
    border: 1px solid rgba(100, 120, 200, 0.3);
    color: #a0aadd;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 11px;
    margin: 2px;
}

/* Explanation box */
.explanation-box {
    background: rgba(201, 168, 76, 0.08);
    border-left: 3px solid #c9a84c;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #d4c4a0;
    margin-top: 12px;
    font-style: italic;
}

/* Pipeline status */
.pipeline-status {
    background: rgba(76, 175, 130, 0.1);
    border: 1px solid rgba(76, 175, 130, 0.3);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px;
    color: #6dd5a0;
    margin-bottom: 16px;
}

/* Star rating */
.stars { color: #c9a84c; font-size: 14px; }

/* Section header */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #e8dcc8;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d2845;
}

/* Agent pipeline visual */
.agent-step {
    display: inline-block;
    background: #1e1a2e;
    border: 1px solid #3d3560;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    color: #a090cc;
    margin: 4px;
}
.agent-step.active { border-color: #c9a84c; color: #c9a84c; background: rgba(201,168,76,0.1); }

/* Hide Streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #a8883a);
    color: #0a0a0f;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.5px;
    padding: 10px 28px;
    width: 100%;
    font-size: 15px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: #1a1828 !important;
    border: 1px solid #2d2845 !important;
    color: #e8dcc8 !important;
    border-radius: 8px !important;
}

.stMultiSelect > div {
    background: #1a1828 !important;
    border: 1px solid #2d2845 !important;
    border-radius: 8px !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #13121c;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #888;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #1e1a2e !important;
    color: #c9a84c !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def star_html(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return (
        '<span class="stars">'
        + "★" * full
        + ("½" if half else "")
        + "☆" * empty
        + f" {rating:.1f}"
        + "</span>"
    )


def sentiment_bars(s: dict) -> str:
    pos = s["positive"] * 100
    neu = s["neutral"] * 100
    neg = s["negative"] * 100
    return f"""
    <div style="font-size:11px;color:#888;margin-bottom:2px;">
        Positive ({pos:.0f}%)
    </div>
    <div class="sentiment-bar-container">
        <div class="sentiment-bar-pos" style="width:{pos:.0f}%"></div>
    </div>
    <div style="font-size:11px;color:#888;margin-bottom:2px;margin-top:5px;">
        Neutral ({neu:.0f}%)
    </div>
    <div class="sentiment-bar-container">
        <div class="sentiment-bar-neu" style="width:{neu:.0f}%"></div>
    </div>
    <div style="font-size:11px;color:#888;margin-bottom:2px;margin-top:5px;">
        Negative ({neg:.0f}%)
    </div>
    <div class="sentiment-bar-container">
        <div class="sentiment-bar-neg" style="width:{neg:.0f}%"></div>
    </div>
    """


def amenity_chips(amenities: list) -> str:
    icons = {
        "wifi": "📶", "pool": "🏊", "spa": "💆", "gym": "🏋️",
        "restaurant": "🍽️", "bar": "🍸", "parking": "🅿️",
        "breakfast": "🥐", "beach": "🏖️", "conference": "📋",
        "yoga": "🧘", "concierge": "🛎️",
    }
    chips = ""
    for a in amenities:
        icon = icons.get(a.lower(), "✓")
        chips += f'<span class="amenity-chip">{icon} {a}</span>'
    return chips


def render_hotel_card(rec: dict, idx: int):
    rank = rec["rank"]
    rank_labels = {1: "🥇 TOP PICK", 2: "🥈 RUNNER UP", 3: "🥉 GREAT CHOICE"}
    rank_label = rank_labels.get(rank, f"#{rank} RECOMMENDED")

    col1, col2 = st.columns([1, 2])

    with col1:
        if rec.get("image_url"):
            st.image(rec["image_url"], use_column_width=True)
        else:
            st.markdown(
                '<div style="background:#1a1828;border-radius:12px;height:180px;'
                'display:flex;align-items:center;justify-content:center;'
                'font-size:48px;">🏨</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="rank-badge">{rank_label}</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:center;font-size:22px;font-weight:700;'
            f'color:#c9a84c;">₹{rec["price_per_night"]:,.0f}<span style="font-size:12px;'
            f'color:#888;">/night</span></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div style="font-family:Playfair Display,serif;font-size:20px;'
            f'font-weight:700;color:#e8dcc8;margin-bottom:4px;">'
            f'{rec["name"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:#888;font-size:13px;margin-bottom:8px;">'
            f'📍 {rec["location"]}  •  {star_html(rec["rating"])}  •  '
            f'{rec["review_count"]:,} reviews</div>',
            unsafe_allow_html=True,
        )
        st.markdown(amenity_chips(rec["amenities"]), unsafe_allow_html=True)

        # Scores row
        st.markdown(
            f'<div style="margin-top:10px;">'
            f'<span class="score-pill">🎯 Score: {rec["comparison_score"]:.3f}</span>'
            f'<span class="score-pill">💬 Sentiment: {rec["sentiment"]["score"]:.2f}</span>'
            f'<span class="score-pill">💰 Value: {rec["price_score"]:.2f}</span>'
            f'<span class="score-pill">🏷️ Amenities: {rec["amenity_score"]*100:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Sentiment bars
        st.markdown(sentiment_bars(rec["sentiment"]), unsafe_allow_html=True)

        # Explanation
        st.markdown(
            f'<div class="explanation-box">💡 {rec["explanation"]}</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar_form():
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px;">
            <div style="font-family:'Playfair Display',serif;font-size:28px;
            color:#c9a84c;font-weight:700;">🏨 SmartHotel</div>
            <div style="font-size:11px;color:#666;letter-spacing:2px;
            text-transform:uppercase;margin-top:4px;">AI Agent System</div>
        </div>
        <hr style="border-color:#2d2845;margin:10px 0 20px;">
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("#### 🗺️ Destination")
    destination = st.sidebar.text_input(
        "City", value="Bangalore", label_visibility="collapsed"
    )

    st.sidebar.markdown("#### 📅 Dates")
    col_a, col_b = st.sidebar.columns(2)
    today = date.today()
    checkin = col_a.date_input("Check-in", value=today + timedelta(days=7), label_visibility="visible")
    checkout = col_b.date_input("Check-out", value=today + timedelta(days=9), label_visibility="visible")

    st.sidebar.markdown("#### 💰 Budget per night (₹)")
    budget = st.sidebar.number_input(
        "Budget", min_value=500, max_value=50000, value=3500, step=500,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("#### 🛏️ Room Type")
    room_type = st.sidebar.selectbox(
        "Room", ["standard", "deluxe", "suite", "studio"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("#### ✈️ Trip Purpose")
    trip_purpose = st.sidebar.selectbox(
        "Purpose", ["leisure", "business", "family", "romantic"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("#### 🏖️ Required Amenities")
    amenities = st.sidebar.multiselect(
        "Amenities",
        ["wifi", "pool", "spa", "gym", "restaurant", "bar",
         "parking", "breakfast", "beach", "conference", "yoga", "concierge"],
        default=["wifi", "pool"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.sidebar.button("🔍  Find Hotels", use_container_width=True)

    return {
        "destination": destination,
        "checkin": str(checkin),
        "checkout": str(checkout),
        "budget_per_night": budget,
        "room_type": room_type,
        "trip_purpose": trip_purpose,
        "amenities": amenities,
        "search": search_btn,
    }


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    form = sidebar_form()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:40px 0 20px;">
            <div style="font-family:'Playfair Display',serif;font-size:48px;
            font-weight:700;background:linear-gradient(135deg,#c9a84c,#e8c96d);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                SmartHotel Agent
            </div>
            <div style="font-size:15px;color:#888;margin-top:8px;letter-spacing:0.5px;">
                Multi-Agent AI System for Intelligent Hotel Discovery & Recommendation
            </div>
            <div style="font-size:11px;color:#555;margin-top:4px;letter-spacing:1.5px;
            text-transform:uppercase;">
                BERT Sentiment · LangChain Orchestration · Real-time Data · Explainable AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline visualisation ─────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:30px;">
            <span class="agent-step active">① Search</span>
            <span style="color:#3d3560;">→</span>
            <span class="agent-step active">② Data Processing</span>
            <span style="color:#3d3560;">→</span>
            <span class="agent-step active">③ BERT Sentiment</span>
            <span style="color:#3d3560;">→</span>
            <span class="agent-step active">④ Comparison</span>
            <span style="color:#3d3560;">→</span>
            <span class="agent-step active">⑤ Recommendation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Agent status check ────────────────────────────────────────────────────
    try:
        status_resp = requests.get(f"{BACKEND_URL}/agents/status", timeout=5)
        if status_resp.ok:
            status_data = status_resp.json()
            agents = status_data.get("agents", {})
            bert_mode = agents.get("3_sentiment_analysis", {}).get("mode", "")
            data_mode = agents.get("1_hotel_search", {}).get("mode", "")
            st.markdown(
                f'<div class="pipeline-status">✅ All 5 agents online  '
                f'│  Data: {data_mode}  │  NLP: {bert_mode}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<div style="background:rgba(224,82,82,0.1);border:1px solid rgba(224,82,82,0.3);'
            'border-radius:8px;padding:10px 16px;font-size:12px;color:#f47878;margin-bottom:16px;">'
            '⚠️ Backend not reachable. Start it with: <code>python backend/main.py</code></div>',
            unsafe_allow_html=True,
        )

    # ── Search ────────────────────────────────────────────────────────────────
    if form["search"]:
        if not form["destination"].strip():
            st.error("Please enter a destination city.")
            return

        with st.spinner("🤖 Running 5-agent pipeline…"):
            payload = {
                "destination": form["destination"],
                "checkin": form["checkin"],
                "checkout": form["checkout"],
                "budget_per_night": form["budget_per_night"],
                "amenities": form["amenities"],
                "room_type": form["room_type"],
                "trip_purpose": form["trip_purpose"],
            }
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the backend. "
                    "Make sure it is running: `python backend/main.py`"
                )
                return
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return

        # ── Results header ────────────────────────────────────────────────────
        recs = data["recommendations"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏙️ City", data["city"])
        col2.metric("🏨 Hotels Found", data["total_recommendations"])
        col3.metric("⏱️ Response Time", f"{data['elapsed_seconds']:.2f}s")
        col4.metric("💰 Budget", f"₹{data['query_budget']:,.0f}/night")

        st.markdown(
            f'<div class="pipeline-status" style="margin-top:16px;">'
            f'ℹ️ {data["pipeline_note"]}</div>',
            unsafe_allow_html=True,
        )

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🏆 Recommendations", "📊 Analytics", "🔍 Raw Data"
        ])

        with tab1:
            st.markdown(
                f'<div class="section-header">'
                f'Top {len(recs)} Hotels in {data["city"]}</div>',
                unsafe_allow_html=True,
            )
            for idx, rec in enumerate(recs):
                with st.container():
                    st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
                    render_hotel_card(rec, idx)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        with tab2:
            st.markdown(
                '<div class="section-header">Analytics Dashboard</div>',
                unsafe_allow_html=True,
            )

            import pandas as pd

            names = [r["name"][:25] for r in recs]

            # ── Scoring Breakdown ─────────────────────────────────────────
            st.markdown("##### 🎯 Scoring Breakdown")
            df_scores = pd.DataFrame({
                "Hotel": names,
                "Composite Score": [round(r["comparison_score"], 3) for r in recs],
                "Sentiment Score": [round(r["sentiment"]["score"], 3) for r in recs],
                "Value Score":     [round(r["price_score"], 3) for r in recs],
                "Amenity Match":   [round(r["amenity_score"], 3) for r in recs],
            })
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

            # ── Sentiment Distribution ────────────────────────────────────
            st.markdown("##### 💬 Sentiment Distribution")
            df_sent = pd.DataFrame({
                "Hotel":    names,
                "Positive": [round(r["sentiment"]["positive"] * 100, 1) for r in recs],
                "Neutral":  [round(r["sentiment"]["neutral"]  * 100, 1) for r in recs],
                "Negative": [round(r["sentiment"]["negative"] * 100, 1) for r in recs],
            })
            st.dataframe(df_sent, use_container_width=True, hide_index=True)

            # ── Price vs Score ────────────────────────────────────────────
            st.markdown("##### 💰 Price vs Composite Score")
            df_price = pd.DataFrame({
                "Hotel":           names,
                "Price (₹/night)": [r["price_per_night"] for r in recs],
                "Rating":          [r["rating"] for r in recs],
                "Composite Score": [round(r["comparison_score"], 3) for r in recs],
            })
            st.dataframe(df_price, use_container_width=True, hide_index=True)

            # ── Visual score bars using progress ──────────────────────────
            st.markdown("##### 📊 Score Visualisation")
            for r in recs:
                st.markdown(f"**{r['name'][:30]}**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Composite", f"{r['comparison_score']:.3f}")
                col2.metric("Sentiment", f"{r['sentiment']['score']:.3f}")
                col3.metric("Value",     f"{r['price_score']:.3f}")
                col4.metric("Amenity",   f"{r['amenity_score']:.0%}")
                st.progress(float(r["comparison_score"]))
                st.markdown("---")

        with tab3:
            st.markdown(
                '<div class="section-header">Raw Pipeline Output</div>',
                unsafe_allow_html=True,
            )
            st.json(data)

    else:
        # Landing state
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;color:#444;">
                <div style="font-size:72px;margin-bottom:20px;">🏨</div>
                <div style="font-size:18px;color:#666;margin-bottom:12px;">
                    Configure your search in the sidebar and click
                    <strong style="color:#c9a84c;">Find Hotels</strong>
                </div>
                <div style="font-size:13px;color:#444;max-width:500px;margin:0 auto;">
                    The 5-agent AI pipeline will search hotels, analyse sentiments
                    with BERT, score with composite C(h) formula and return
                    explainable, personalised recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show architecture info
        with st.expander("📐 System Architecture", expanded=False):
            st.markdown("""
**Five-Agent Pipeline (as per research paper):**

| Agent | Role | Technology |
|-------|------|-----------|
| ① Hotel Search | Fetch real-time hotel data | SerpAPI / Google Hotels |
| ② Data Processing | Clean, tokenise, TF-IDF features | NLTK, scikit-learn |
| ③ Sentiment Analysis | Classify reviews Pos/Neu/Neg | BERT (bert-base-uncased) |
| ④ Comparison | Compute composite score C(h) | Weighted formula |
| ⑤ Recommendation | Rank top-K + explain | LangChain orchestration |

**Composite Score Formula:**
```
C(h) = 0.25·Price + 0.30·Rating + 0.30·Sentiment + 0.15·Amenity
```

**Sentiment Score Formula:**
```
S = (0.6 × P) + (0.2 × Ne) − (0.6 × Neg)
```
            """)


if __name__ == "__main__":
    main()
