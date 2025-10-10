import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from numpy.linalg import norm
from sklearn.preprocessing import normalize

try:
    from implicit.als import AlternatingLeastSquares
    als_available = True
except Exception:
    als_available = False

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

ratings['interaction'] = (ratings['rating'] >= 4).astype(int)
interactions = ratings[ratings['interaction'] > 0].copy()

user_map = {u: i for i, u in enumerate(interactions['userId'].unique())}
item_map = {i: j for j, i in enumerate(interactions['movieId'].unique())}
inv_user_map = {v: k for k, v in user_map.items()}
inv_item_map = {v: k for k, v in item_map.items()}

interactions['u_idx'] = interactions['userId'].map(user_map)
interactions['i_idx'] = interactions['movieId'].map(item_map)

rows = interactions['i_idx'].astype(int)
cols = interactions['u_idx'].astype(int)
data = np.ones(len(interactions), dtype=np.float32)

num_users = len(user_map)
num_items = len(item_map)
sparse_item_user = csr_matrix((data, (rows, cols)), shape=(num_items, num_users))

if als_available:
    als_model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=20, use_gpu=False)
    als_model.fit(sparse_item_user * 1.0)
    item_factors = als_model.item_factors
    user_factors = als_model.user_factors
    item_factors_norm = item_factors / (np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-9)
    user_factors_norm = user_factors / (np.linalg.norm(user_factors, axis=1, keepdims=True) + 1e-9)
else:
    als_model = None
    item_factors_norm = np.zeros((num_items, 64))
    user_factors_norm = np.zeros((num_users, 64))

movies_sub = movies[movies['movieId'].isin(item_map.keys())].copy()
movies_sub['i_idx'] = movies_sub['movieId'].map(item_map)
movies_sub['text'] = (movies_sub['title'].fillna('') + ' ' + movies_sub['genres'].fillna('')).astype(str)

tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = tf.fit_transform(movies_sub['text'])
tfidf_index = {row['i_idx']: idx for idx, row in movies_sub.reset_index().iterrows()}
tfidf_norm = normalize(tfidf_matrix, axis=1)

def recommend_candidates_for_user(user_id, topk_collab=100, topk_content=100):
    if user_id not in user_map:
        return []
    uidx = user_map[user_id]
    collab_items = []
    if 'als_model' in globals() and als_available:
        try:
            collab = als_model.recommend(userid=uidx, user_items=None, N=topk_collab, filter_already_liked_items=True)
            collab_items = [i for i, score in collab]
        except Exception:
            collab_items = []
    content_items = []
    hist = interactions[interactions['u_idx'] == uidx]['i_idx'].unique()
    if len(hist) > 0:
        hist_indices = [tfidf_index[i] for i in hist if i in tfidf_index]
        if hist_indices:
            hist_vec = np.asarray(tfidf_matrix[hist_indices].mean(axis=0)).ravel()
            sims = cosine_similarity(hist_vec.reshape(1, -1), tfidf_matrix).ravel()
            top_idxs = np.argpartition(-sims, min(topk_content, len(sims)))[:topk_content]
            content_items = [movies_sub.iloc[i]['i_idx'] for i in top_idxs]
    candidates = list(dict.fromkeys(collab_items + content_items))
    return candidates

ratings_sorted = ratings.sort_values(['userId','timestamp'])
def leave_one_out(df):
    train_list, test_list = [], []
    for user, g in df.groupby('userId'):
        if len(g) < 2:
            train_list.append(g)
            continue
        train_list.append(g.iloc[:-1])
        test_list.append(g.iloc[-1:])
    return pd.concat(train_list), pd.concat(test_list) if test_list else (pd.concat(train_list), pd.DataFrame())

train_ratings, test_ratings = leave_one_out(ratings_sorted)
user_train_items = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()
item_pop = train_ratings['movieId'].value_counts().to_dict()
movieid_to_iidx = {m: item_map[m] for m in item_map}

def build_feature_row(user_id, item_id):
    features = {}
    uid_present = user_id in user_map
    iid_present = item_id in item_map
    if 'als_model' in globals() and uid_present and iid_present:
        try:
            uidx = user_map[user_id]
            iidx = item_map[item_id]
            features['collab_score'] = float(np.dot(user_factors_norm[uidx], item_factors_norm[iidx]))
        except Exception:
            features['collab_score'] = 0.0
    else:
        features['collab_score'] = 0.0
    content_sim = 0.0
    if uid_present and iid_present:
        hist = user_train_items.get(user_id, set())
        hist_iidx = [movieid_to_iidx[i] for i in hist if i in movieid_to_iidx]
        hist_rows = [tfidf_index[i] for i in hist_iidx if i in tfidf_index and tfidf_index[i] < tfidf_matrix.shape[0]]
        item_row = tfidf_index.get(item_map[item_id], None)
        if hist_rows and item_row is not None and item_row < tfidf_matrix.shape[0]:
            user_vec = np.asarray(tfidf_matrix[hist_rows].mean(axis=0)).ravel()
            item_vec = np.asarray(tfidf_matrix[item_row].todense()).ravel()
            denom = (norm(user_vec) * norm(item_vec) + 1e-9)
            content_sim = float(np.dot(user_vec, item_vec) / denom)
    features['content_sim'] = content_sim
    features['popularity'] = float(item_pop.get(item_id, 0))
    features['is_popular_top100'] = int(item_pop.get(item_id, 0) > 100)
    return features

train_rows = []
for _, row in test_ratings.iterrows():
    u = row['userId']; pos_item = row['movieId']
    pos_feats = build_feature_row(u, pos_item)
    pos_feats['label'] = 1
    pos_feats['userId'] = u
    pos_feats['movieId'] = pos_item
    train_rows.append(pos_feats)
    sampled = 0
    popular_items = list(item_pop.keys())[:1000]
    for neg_item in popular_items:
        if neg_item in user_train_items.get(u,set()) or neg_item == pos_item:
            continue
        neg_feats = build_feature_row(u, neg_item)
        neg_feats['label'] = 0
        neg_feats['userId'] = u
        neg_feats['movieId'] = neg_item
        train_rows.append(neg_feats)
        sampled += 1
        if sampled >= 10: break

train_df = pd.DataFrame(train_rows).fillna(0)
FEATURE_COLS = ['collab_score','content_sim','popularity','is_popular_top100']
X, y = train_df[FEATURE_COLS], train_df['label']
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
params = {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':31,'seed':42,'verbosity':-1}
callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
gbm = lgb.train(params, dtrain, valid_sets=[dtrain, dval], num_boost_round=1000, callbacks=callbacks)
import os
os.makedirs('models', exist_ok=True)
joblib.dump(gbm, 'models/lgb_ranker.joblib')
if 'als_model' in globals():
    joblib.dump(als_model, 'models/als_model.joblib')
joblib.dump(tf, 'models/tfidf.joblib')
joblib.dump(movies_sub[['movieId','i_idx','text']].set_index('movieId'), 'models/movies_index.joblib')
joblib.dump(user_map, 'models/user_map.joblib')
joblib.dump(item_map, 'models/item_map.joblib')

def recommend_for_user_api(user_id, k=10):
    candidates = recommend_candidates_for_user(user_id, topk_collab=200, topk_content=200)
    rows = []
    for cid in candidates:
        itemid = inv_item_map[cid]
        feats = build_feature_row(user_id, itemid)
        rows.append({'movieId': itemid, **feats})
    dfc = pd.DataFrame(rows)
    dfc['score'] = gbm.predict(dfc[FEATURE_COLS])
    return dfc.sort_values('score', ascending=False).head(k)[['movieId','score']].to_dict(orient='records')

sample_users = test_ratings['userId'].unique()[:5]
for u in sample_users:
    print(f"User {u}: {recommend_for_user_api(u,k=10)}")
