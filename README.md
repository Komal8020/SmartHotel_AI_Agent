# SmartHotel Agent 🏨

**A Multi-Agent AI System for Intelligent Hotel Discovery and Recommendation**

Based on the research paper: *"SmartHotel Agent: An Agentic AI System for Intelligent Hotel Discovery and Recommendation in Unknown Locations"* (CIE3 Research Paper, RV University Bengaluru)

---

## 📐 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangChain Orchestrator                        │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐  │
│  │ Agent 1  │→ │  Agent 2     │→ │  Agent 3  │→ │ Agent 4  │  │
│  │  Hotel   │  │    Data      │  │ Sentiment │  │Comparison│  │
│  │  Search  │  │ Processing   │  │  (BERT)   │  │  C(h)    │  │
│  │ SerpAPI  │  │ NLTK+TF-IDF  │  │89.2% acc. │  │ Weights  │  │
│  └──────────┘  └──────────────┘  └───────────┘  └──────────┘  │
│                                                         │        │
│                                                    ┌────▼─────┐  │
│                                                    │ Agent 5  │  │
│                                                    │  Rec.    │  │
│                                                    │  Top-K   │  │
│                                                    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────┐     ┌────────────────────────────┐
│   FastAPI Backend          │     │   Streamlit Frontend       │
│   POST /recommend          │────▶│   Luxury dark UI           │
│   GET  /agents/status      │     │   Charts + Explanations    │
└────────────────────────────┘     └────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd smarthotel_agent
pip install -r requirements.txt
```

### 2. Configure API keys (optional but recommended)

```bash
cp .env.example .env
# Edit .env and add your SerpAPI key:
# SERPAPI_KEY=your_key_here
```
> **Without a SerpAPI key:** The app runs in demo mode with 8 realistic hotel profiles.  
> Get a free key at https://serpapi.com (100 free searches/month)

### 3. Start the FastAPI backend

```bash
python backend/main.py
# OR
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at: http://localhost:8000  
API docs at: http://localhost:8000/docs

### 4. Start the Streamlit frontend

```bash
# In a second terminal:
python -m streamlit run frontend/app.py
```

Frontend opens at: http://localhost:8501

---

## 🔌 API Reference

### POST `/recommend`

```json
{
  "destination": "Bangalore",
  "checkin": "2025-08-01",
  "checkout": "2025-08-03",
  "budget_per_night": 3500,
  "amenities": ["wifi", "pool", "gym"],
  "room_type": "deluxe",
  "trip_purpose": "leisure"
}
```

**Response:**
```json
{
  "city": "Bangalore",
  "query_budget": 3500,
  "elapsed_seconds": 2.8,
  "total_recommendations": 5,
  "recommendations": [
    {
      "rank": 1,
      "name": "Heritage Boutique Hotel",
      "price_per_night": 4200,
      "rating": 4.6,
      "sentiment": { "positive": 0.80, "neutral": 0.15, "negative": 0.05, "score": 0.45 },
      "comparison_score": 0.812,
      "explanation": "Ranked #1 with a sentiment score of 0.45, ₹700 above your budget..."
    }
  ]
}
```

### GET `/agents/status`
Returns status of all 5 agents and their active modes.

---

## 📊 Scoring Formulas (from paper)

### Composite Score C(h)
```
C(h) = 0.25 × Price + 0.30 × Rating + 0.30 × Sentiment + 0.15 × Amenity
```

### Sentiment Score S
```
S = (0.6 × P) + (0.2 × Ne) - (0.6 × Neg)
```
where P = proportion positive, Ne = neutral, Neg = negative reviews

---

## 📁 Project Structure

```
smarthotel_agent/
├── config.py                     # Global config, weights, data models
├── orchestrator.py               # LangChain 5-agent pipeline coordinator
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── search_agent.py           # Agent 1: SerpAPI hotel search
│   ├── data_processing_agent.py  # Agent 2: NLTK + TF-IDF preprocessing  
│   ├── sentiment_agent.py        # Agent 3: BERT sentiment (bert-base-uncased)
│   ├── comparison_agent.py       # Agent 4: Composite C(h) scoring
│   └── recommendation_agent.py   # Agent 5: Top-K + NL explanations
│
├── backend/
│   └── main.py                   # FastAPI REST API
│
└── frontend/
    └── app.py                    # Streamlit web UI
```

---

## 🔧 Performance Results (from paper)

| Metric | SmartHotel AI | Baseline B1 | Baseline B2 |
|--------|--------------|-------------|-------------|
| Sentiment Accuracy | **89.2%** | N/A | N/A |
| Top-5 Rec. Accuracy | **87.6%** | 71.3% | 68.9% |
| Personalization P@10 | **0.84** | 0.61 | 0.58 |
| Avg Response Time | 2.8s | 1.1s | 0.9s |
| User Satisfaction | **4.3/5** | 3.1 | 2.9 |

---

## 🔮 Future Work (from paper)

- [ ] Switch to LLaMA-2 / Mistral-7B for explanation generation
- [ ] Feedback loop for continual weight learning (w1–w4)
- [ ] Multi-language support via mBERT/XLM-RoBERTa
- [ ] Mobile app with voice commands
- [ ] ARIMA/Transformer price prediction
- [ ] Payment gateway integration

---

## 📚 Key References

- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
- LangChain Framework — Agent orchestration
- SerpAPI — Google Hotels real-time data
- TripAdvisor Dataset (142k reviews) — Sentiment training data
