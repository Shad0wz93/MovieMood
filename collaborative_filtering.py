import pandas as pd
from utils.post_processing import post_processing
from utils.pre_processing import pre_processing
from utils.processing import processing

# On récupère les données dans les fichiers csv
movies = pd.read_csv('datas/tmdb_5000_movies.csv')
ratings = pd.read_csv('datas/ratings_small.csv')

# On veut faire l'analyse pour l'utilisateur 2
user_id = 2

(matrix_csr, user_index, item_ids) = pre_processing(movies, ratings, user_id)


# La magie
ids, scores = processing(matrix_csr, user_index)
(recommended_df, user_ratings_sorted) =  post_processing(movies, ratings, item_ids, ids, scores, user_id)
recommended_df.to_csv('recommended_movies.csv')


# --- Print results ---
print(f"\n🎬 Movies rated by user {user_id}:\n")
print(user_ratings_sorted.head(10).to_string(index=False))

print(f"\n⭐ Top recommendations for user {user_id}:\n")
print(recommended_df.head(10).to_string(index=False))