from src.data_preprocessing import load_and_process_movies
from src.content_based import build_content_model
from src.collaborative import build_als_model
from src.hybrid_adaptive import get_adaptive_recommendations
import pandas as pd
from scipy.sparse import coo_matrix

# -------------------------------
# 1️⃣ Charger les données
# -------------------------------
movies = load_and_process_movies(
    'data/tmdb_5000_movies.csv',
    'data/tmdb_5000_credits.csv'
)

ratings = pd.read_csv('data/ratings_small.csv')[['userId','movieId','rating','timestamp']]

# -------------------------------
# 2️⃣ Construire les modèles
# -------------------------------

# Content-based (similarité cosine)
cosine_sim, indices = build_content_model(movies)

# ALS / filtrage collaboratif
als_model, ratings = build_als_model(ratings)

# Créer la matrice CSR pour ALS
ratings_csr = coo_matrix(
    (ratings['rating'],
     (ratings['userId_cat'].cat.codes, ratings['movieId_cat'].cat.codes))
).tocsr()

# -------------------------------
# 3️⃣ Recommandations adaptatives
# -------------------------------
user_id = 2  # changer pour tester un autre utilisateur

recommendations = get_adaptive_recommendations(
    user_id=user_id,
    movies=movies,
    ratings=ratings,
    indices=indices,
    cosine_sim=cosine_sim,
    model=als_model,
    ratings_csr=ratings_csr,
    top_n=10
)

# -------------------------------
# 4️⃣ Afficher les résultats
# -------------------------------
print(f"\n🎬 Recommandations hybrides adaptatives pour l'utilisateur {user_id} :\n")
print(recommendations.to_string(index=False))
