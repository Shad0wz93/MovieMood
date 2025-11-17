import pandas as pd

def post_processing(movies, ratings, item_ids, ids, scores, user_id):
    # Mapping des IDs internes vers les movieId
    movie_labels = dict(enumerate(item_ids.cat.categories))
    recommended_movies = [(movie_labels[iid], float(score)) for iid, score in zip(ids, scores)]

    # Notes existantes de l'utilisateur
    user_ratings = ratings[ratings['userId'] == user_id].merge(
        movies, left_on='movieId', right_on='id', how='left'
    )[['title', 'release_date', 'rating']].dropna()
    user_ratings_sorted = user_ratings.sort_values(by='rating', ascending=False)

    # Préparation du DataFrame final des recommandations
    recommended_df = pd.DataFrame(recommended_movies, columns=['movieId', 'score']).merge(
        movies, left_on='movieId', right_on='id', how='left'
    )[['title', 'release_date', 'score']].dropna()

    return recommended_df, user_ratings_sorted
