import pandas as pd
import numpy as np

def weighted_score(row, C=6.0, m=50):
    """Score IMDb pondéré pour popularité et note"""
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

def like_dislike(rating):
    """Convertit une note en like/dislike"""
    if rating > 3:
        return 1  # Like
    else:
        return -1  # Dislike

def get_adaptive_recommendations(user_id, movies, ratings, indices, cosine_sim, model, ratings_csr, top_n=10,
                                 current_date=None):
    """
    Retourne des recommandations hybrides adaptatives pour un utilisateur.
    """
    if current_date is None:
        current_date = pd.Timestamp.now()

    # Assurer timestamp en datetime
    if 'timestamp' in ratings.columns and not np.issubdtype(ratings['timestamp'].dtype, np.datetime64):
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])

    # Ajouter colonne like/dislike
    ratings['like_dislike'] = ratings['rating'].apply(like_dislike)

    # Pondération temporelle : plus récent = plus important
    if 'timestamp' in ratings.columns:
        ratings['time_weight'] = np.exp(-(current_date - ratings['timestamp']).dt.days / 365)
    else:
        ratings['time_weight'] = 1.0

    # Films vus par l'utilisateur
    user_ratings = ratings[ratings['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['like_dislike'] > 0]['movieId'].tolist()
    disliked_movies = user_ratings[user_ratings['like_dislike'] < 0]['movieId'].tolist()

    # -----------------------------
    # 1️⃣ Content-based (positif et négatif)
    # -----------------------------
    content_scores = {}

    # Likes → augmenter similarité
    for mid in liked_movies:
        title = movies[movies['id'] == mid]['title'].values
        if len(title) == 0:
            continue
        idx = indices[title[0]]
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i, score in sim_scores:
            movie_id = movies.iloc[i]['id']
            if movie_id not in liked_movies and movie_id not in disliked_movies:
                weight = user_ratings[user_ratings['movieId'] == mid]['time_weight'].values[0]
                content_scores[movie_id] = content_scores.get(movie_id, 0) + score * weight

    # Dislikes → diminuer similarité
    for mid in disliked_movies:
        title = movies[movies['id'] == mid]['title'].values
        if len(title) == 0:
            continue
        idx = indices[title[0]]
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i, score in sim_scores:
            movie_id = movies.iloc[i]['id']
            if movie_id not in liked_movies and movie_id not in disliked_movies:
                weight = user_ratings[user_ratings['movieId'] == mid]['time_weight'].values[0]
                content_scores[movie_id] = content_scores.get(movie_id, 0) - score * weight

    # -----------------------------
    # 2️⃣ ALS / filtrage collaboratif
    # -----------------------------
    try:
        user_index = ratings['userId_cat'].cat.categories.get_loc(user_id)
        ids, scores = model.recommend(user_index, ratings_csr[user_index])
        als_scores = {ratings['movieId_cat'].cat.categories[i]: score for i, score in zip(ids, scores)}
    except KeyError:
        als_scores = {}

    # -----------------------------
    # 3️⃣ Combinaison scores
    # -----------------------------
    hybrid_scores = {}
    for movie_id in set(list(content_scores.keys()) + list(als_scores.keys())):
        hybrid_scores[movie_id] = 0.5 * content_scores.get(movie_id, 0) + 0.5 * als_scores.get(movie_id, 0)

    # -----------------------------
    # 4️⃣ DataFrame final avec pondération IMDb
    # -----------------------------
    recommended_movies = movies[movies['id'].isin(hybrid_scores.keys())][['title', 'vote_average', 'vote_count', 'id']].copy()
    recommended_movies['hybrid_score'] = recommended_movies['id'].map(hybrid_scores)
    recommended_movies['weighted_score'] = recommended_movies.apply(weighted_score, axis=1)

    # Filtrer films trop peu notés
    recommended_movies = recommended_movies[recommended_movies['vote_count'] >= 10]

    # Score final combiné
    recommended_movies['final_score'] = recommended_movies['hybrid_score'] * recommended_movies['weighted_score']

    # -----------------------------
    # 5️⃣ Filtrage films déjà notés
    # -----------------------------
    rated_movie_ids = user_ratings['movieId'].unique()
    recommended_movies = recommended_movies[~recommended_movies['id'].isin(rated_movie_ids)]

    # Trier et retourner top N
    recommended_movies = recommended_movies.sort_values('final_score', ascending=False)

    return recommended_movies.head(top_n)[['title', 'vote_average', 'vote_count', 'id']]