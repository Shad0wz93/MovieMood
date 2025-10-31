import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

# On récupère les données dans les fichiers csv
movies = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')

# Pour le rendu
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

# Tranforme les user et movies en id uniques
user_ids = ratings["userId"].astype('category')
item_ids = ratings["movieId"].astype('category')

# Créé une matrice pour stocker toutes notes
# Chaque ligne correspond à un utilisateur
# Chaque colonne correspond à un film
matrix = coo_matrix((ratings['rating'], (user_ids.cat.codes, item_ids.cat.codes)))

# Convertit la matrice en csr (compressed sparse row) pour faciliter la recherche (et rend compatible la fonction utilisée après)
matrix_csr = matrix.tocsr()

# La magie
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15)
model.fit(matrix_csr)

# On veut faire l'analyse pour l'utilisateur 2
user_id = 2

# On récupère l'index de cet utilisateur dans la liste de tous les user_ids
user_index = user_ids.cat.categories.get_loc(user_id)

# La magie
ids, scores = model.recommend(user_index, matrix_csr[user_index])

movie_labels = dict(enumerate(item_ids.cat.categories))
recommended_movies = [(movie_labels[iid], float(score)) for iid, score in zip(ids, scores)]

user_ratings = ratings[ratings['userId'] == user_id].merge(
    movies, left_on='movieId', right_on='id', how='left'
)[['title', 'release_date', 'rating']].dropna()

user_ratings_sorted = user_ratings.sort_values(by='rating', ascending=False)

# --- Show recommended movies with titles ---
recommended_df = pd.DataFrame(recommended_movies, columns=['movieId', 'score']).merge(
    movies, left_on='movieId', right_on='id', how='left'
)[['title', 'release_date', 'score']].dropna()

# --- Print results ---
print(f"\n🎬 Movies rated by user {user_id}:\n")
print(user_ratings_sorted.head(10).to_string(index=False))

print(f"\n⭐ Top recommendations for user {user_id}:\n")
print(recommended_df.head(10).to_string(index=False))