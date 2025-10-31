import pandas as pd


def weighted_score(row, C=6.0, m=50):
    """
    Score pondéré type IMDb pour un film.
    C : moyenne générale de vote_average
    m : minimum de votes pour être significatif
    """
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


def get_hybrid_recommendations(user_id, movies, ratings, indices, cosine_sim, model, ratings_csr, top_n=10):
    # 1️⃣ Films déjà vus par l'utilisateur
    watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    # 2️⃣ Scores ALS
    user_index = ratings['userId_cat'].cat.categories.get_loc(user_id)
    ids, scores = model.recommend(user_index, ratings_csr[user_index])
    als_recs = {ratings['movieId_cat'].cat.categories[i]: score for i, score in zip(ids, scores)}

    # 3️⃣ Scores contenu
    sim_scores_total = {}
    for mid in watched:
        title = movies[movies['id'] == mid]['title'].values
        if len(title) == 0:
            continue
        idx = indices[title[0]]
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i, score in sim_scores:
            movie_id = movies.iloc[i]['id']
            if movie_id not in watched:
                sim_scores_total[movie_id] = sim_scores_total.get(movie_id, 0) + score

    # 4️⃣ Combinaison scores ALS + contenu
    hybrid_scores = {}
    for movie_id in set(list(als_recs.keys()) + list(sim_scores_total.keys())):
        hybrid_scores[movie_id] = 0.5 * als_recs.get(movie_id, 0) + 0.5 * sim_scores_total.get(movie_id, 0)

    # 5️⃣ Créer DataFrame des recommandations
    recommended_ids = list(hybrid_scores.keys())
    recommended_movies = movies[movies['id'].isin(recommended_ids)][['title', 'vote_average', 'vote_count', 'id']]

    # 6️⃣ Ajouter score hybride et pondération IMDb
    recommended_movies['hybrid_score'] = recommended_movies['id'].map(hybrid_scores)
    recommended_movies['weighted_score'] = recommended_movies.apply(weighted_score, axis=1)

    # 7️⃣ Filtrer films avec trop peu de votes
    recommended_movies = recommended_movies[recommended_movies['vote_count'] >= 10]

    # 8️⃣ Tri par combinaison : hybrid_score * weighted_score
    recommended_movies['final_score'] = recommended_movies['hybrid_score'] * recommended_movies['weighted_score']
    recommended_movies = recommended_movies.sort_values('final_score', ascending=False)

    return recommended_movies.head(top_n)[['title', 'vote_average', 'vote_count']]