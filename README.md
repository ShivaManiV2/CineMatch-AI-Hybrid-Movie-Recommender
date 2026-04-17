# 🎬 CineMatch AI: Hybrid Movie Recommender

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

A production-ready Hybrid Movie Recommendation Engine that combines **Collaborative Filtering** and **Content-Based Filtering** with a **LightGBM Gradient Boosting Ranker**. Featuring a premium glassmorphic dashboard and real-time performance telemetry.

---

## 🚀 Key Features

*   **Hybrid Intelligence**: Blends user-item interactions (ALS) with movie metadata (TF-IDF) using a machine learning ranker.
*   **Zero-Failure Cold Start**: Graceful fallback to global popularity metrics for new users (ensure 100% service availability).
*   **Premium UI**: A high-fidelity, dark-mode dashboard with micro-animations and responsive glassmorphism.
*   **Real-time Telemetry**: Custom ASGI middleware tracks and reports engine latency through `X-Process-Time` headers.
*   **Production-Ready**: Multi-stage Docker containerization and Pydantic-enforced schema validation.

---

## 🛠️ Technology Stack

*   **Core**: Python 3.9+, FastAPI, Pydantic
*   **ML Stack**: Scikit-Learn, LightGBM, Pandas, NumPy, Joblib
*   **Frontend**: HTML5, Vanilla CSS (Glassmorphism), Google Fonts (Outfit)
*   **DevOps**: Docker (Multi-stage builds), Uvicorn

---

## 📂 Project Structure

```text
├── api.py                  # Production FastAPI Application (Models, Schemas, Logic)
├── Dockerfile              # Multi-stage optimized Docker build
├── requirements.txt        # Managed dependencies
├── test_api.py            # Diagnostic script for latency & schema verification
├── data/                   # Dataset storage (movies.csv, ratings.csv)
├── models/                 # Pre-trained ML artifacts & transformers
└── frontend/               
    └── index.html          # Premium Dashboard UI
```

---

## ⚙️ Setup & Installation

### 1. Local Development
```bash
# Clone the repo
git clone <repo-url> && cd project

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn api:app --reload
```

### 2. Docker Deployment (Recommended)
```bash
# Build the production-optimized image
docker build -t cinematch-ai .

# Run the container
docker run -p 8000:8000 cinematch-ai
```

---

## 📡 API Documentation

### POST `/recommend`
Generates personalized rankings for a specific profile.

**Request Body:**
```json
{
  "user_id": 1,
  "num_rec": 10
}
```

**Response Header:**
*   `X-Process-Time`: Measures engine inference speed in seconds.

---

## 📊 Performance & Scalability

| Scenario | Latency Profile | Strategy |
| :--- | :--- | :--- |
| **Cold Start** | `< 10ms` | Precomputed global popularity fallback |
| **Personalized** | `~800ms` | Real-time TF-IDF & LightGBM Ranking |
| **Model Pattern** | **Pre-Loaded** | Models hoisted to global scope for zero-IO inference |

---

## 🎨 UI Dashboard
Access the premium interface at `http://localhost:8000/ui`. The dashboard features:
- **Neon-accented telemetry** showing real-time backend speed.
- **Frosted glass UI** optimized for visual focus.
- **Dynamic animations** that slide results into view.

---

## 📝 License
Distributed under the MIT License.
