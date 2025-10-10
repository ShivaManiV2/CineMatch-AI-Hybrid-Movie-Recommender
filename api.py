# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles





# Load saved models and artifacts

gbm = joblib.load('models/lgb_ranker.joblib')
user_map = joblib.load('models/user_map.joblib')
item_map = joblib.load('models/item_map.joblib')
movies_index = joblib.load('models/movies_index.joblib')
tfidf = joblib.load('models/tfidf.joblib')

inv_item_map = {v: k for k, v in item_map.items()}

# Load interactions if needed
ratings = pd.read_csv("data/ratings.csv")  # adjust path

# Precompute TF-IDF matrix
movies_sub = movies_index.reset_index()
tfidf_matrix = tfidf.transform(movies_sub['text'])
tfidf_matrix = normalize(tfidf_matrix, axis=1)
tfidf_index = {row['i_idx']: idx for idx, row in movies_sub.reset_index().iterrows()}



# FastAPI setup

app = FastAPI()

class RecRequest(BaseModel):
    user_id: int
    k: int = 10


# Helper functions

def recommend_candidates_for_user(user_id, topk_content=100):
    if user_id not in user_map:
        return []
    uidx = user_map[user_id]
    hist = ratings[ratings['userId'] == user_id]['movieId'].unique()
    # content candidates
    hist_iidx = [i for i in hist if i in item_map]
    hist_rows = [tfidf_index[item_map[i]] for i in hist_iidx if item_map[i] in tfidf_index]
    if not hist_rows:
        return []
    hist_vec = np.asarray(tfidf_matrix[hist_rows].mean(axis=0)).ravel()
    sims = cosine_similarity(hist_vec.reshape(1, -1), tfidf_matrix).ravel()
    top_idxs = np.argpartition(-sims, topk_content)[:topk_content]
    return [movies_sub.iloc[i]['i_idx'] for i in top_idxs]

def build_feature_row(user_id, item_id):
    features = {}
    # collab_score = 0 since ALS not loaded
    features['collab_score'] = 0.0
    # content score
    hist = ratings[ratings['userId'] == user_id]['movieId'].unique()
    hist_iidx = [i for i in hist if i in item_map]
    hist_rows = [tfidf_index[item_map[i]] for i in hist_iidx if item_map[i] in tfidf_index]
    if hist_rows and item_id in item_map and item_map[item_id] in tfidf_index:
        user_vec = np.asarray(tfidf_matrix[hist_rows].mean(axis=0)).ravel()
        item_vec = np.asarray(tfidf_matrix[tfidf_index[item_map[item_id]]].todense()).ravel()
        features['content_sim'] = float(np.dot(user_vec, item_vec)/(np.linalg.norm(user_vec)*np.linalg.norm(item_vec)+1e-9))
    else:
        features['content_sim'] = 0.0
    features['popularity'] = 0  # optional: load from ratings
    features['is_popular_top100'] = 0
    return features

def recommend_for_user_api(user_id, k=10):
    candidates = recommend_candidates_for_user(user_id, topk_content=200)
    rows = []
    for cid in candidates:
        feats = build_feature_row(user_id, cid)
        rows.append({'movieId': cid, **feats})
    dfc = pd.DataFrame(rows)
    if dfc.empty:
        return []
    FEATURE_COLS = ['collab_score','content_sim','popularity','is_popular_top100']
    dfc['score'] = gbm.predict(dfc[FEATURE_COLS])
    dfc = dfc.sort_values('score', ascending=False).head(k)
    return dfc[['movieId','score']].to_dict(orient='records')


# API endpoints

@app.get("/")
def root():
    return {"message": "Recommender API running"}

@app.post("/recommend")
def recommend(req: RecRequest):
    recs = recommend_for_user_api(req.user_id, k=req.k)
    return {"user": req.user_id, "recs": recs}

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()