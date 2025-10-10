Hybrid Movie Recommender System

A hybrid movie recommendation system using content-based and collaborative filtering approaches with a LightGBM ranker, served via FastAPI with a clean frontend for easy interaction.

Features

Hybrid recommendation:

Content-based (TF-IDF on movie titles + genres)

Collaborative filtering (ALS fallback, optional)

LightGBM ranker to combine features

FastAPI backend serving recommendations via REST API

Modern responsive frontend with Bootstrap

Easy to test for any user in the dataset

Project Structure
project/
│
├── api.py                  # FastAPI backend
├── hybrid_recommender_safe.py  # Training script
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── models/                 # Saved models and artifacts
│   ├── lgb_ranker.joblib
│   ├── movies_index.joblib
│   ├── tfidf.joblib
│   ├── user_map.joblib
│   └── item_map.joblib
├── frontend/
│   └── index.html          # Frontend HTML page
└── README.md

Setup Instructions
1. Clone the repository
git clone <your-repo-url>
cd project

2. Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn lightgbm joblib scipy


Optional for ALS: pip install implicit

3. Train the model (if not using pre-trained)
python hybrid_recommender_safe.py


This will create the models/ folder with all required artifacts.

Running the API
uvicorn api:app --reload


The API will be available at http://127.0.0.1:8000.

Endpoints

Root:

GET /


Returns: {"message": "Recommender API running"}

Recommend:

POST /recommend
Content-Type: application/json
Body: { "user_id": <int>, "k": <int> }


Example Response:

{
  "user": 1,
  "recs": [
    {"movieId": 12, "score": 0.876},
    {"movieId": 25, "score": 0.852}
  ]
}

Frontend

Open the frontend in the browser via FastAPI:

Ensure index.html is in frontend/ folder.

Serve it via FastAPI:

from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


Open in browser:

http://127.0.0.1:8000/ui


Enter User ID and Number of Recommendations, click Get Recommendations, and view results.

Notes

user_id must exist in your dataset (ratings.csv) for recommendations to work.

k specifies the number of recommendations to return.

Content-based recommendations use TF-IDF vectors of movie titles and genres.

Collaborative filtering (ALS) is optional; if ALS is not installed, collab_score is set to 0.

Dependencies

Python 3.9+

FastAPI

Uvicorn

Pandas, NumPy

Scikit-learn

LightGBM

Joblib

SciPy

Optional: Implicit (for ALS)

Future Improvements

Display actual movie titles and genres in the frontend instead of movie IDs.

Add user authentication to personalize recommendations.

Use a pre-trained embeddings model (like Sentence-BERT) for richer content features.

Add pagination for large recommendation lists.

License

MIT License
