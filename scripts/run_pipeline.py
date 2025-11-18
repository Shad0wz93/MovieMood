import pandas as pd
import os
import mlflow
from implicit.als import AlternatingLeastSquares
from utils.pre_processing import pre_processing
from utils.processing import processing
from utils.post_processing import post_processing

# Configuration
movies_path = 'tmdb_5000_movies.csv'
ratings_path = 'ratings_small.csv'
credits_path = 'tmdb_5000_credits.csv'

for file_path in [movies_path, ratings_path, credits_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier manquant : {file_path}")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# ---------------------------
# Dossier de sortie
# ---------------------------
os.makedirs('outputs', exist_ok=True)

# ---------------------------
# Paramètres ALS
# ---------------------------
factors = 50
regularization = 0.1
iterations = 15

# ---------------------------
# Entrée utilisateur
# ---------------------------
user_id = int(input("Entrez l'ID de l'utilisateur pour les recommandations : "))

# ---------------------------
# Pipeline
# ---------------------------
matrix_csr, user_index, item_ids = pre_processing(movies, ratings, user_id)
ids, scores = processing(matrix_csr, user_index)
recommended_df, user_ratings_sorted = post_processing(movies, ratings, item_ids, ids, scores, user_id)

# ---------------------------
# MLflow logging
# ---------------------------
with mlflow.start_run():
    mlflow.log_param("factors", factors)
    mlflow.log_param("regularization", regularization)
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("user_id", user_id)

    # Optionnel : log métriques simples
    mlflow.log_metric("num_recommendations", len(recommended_df))

    # Log CSV de recommandations
    output_file = f'outputs/recommended_movies_{user_id}.csv'
    recommended_df.to_csv(output_file, index=False)
    mlflow.log_artifact(output_file)

    print(f"\n✅ Recommandations sauvegardées dans : {output_file}\n")
    print(user_ratings_sorted.head(10).to_string(index=False))
    print(recommended_df.head(10).to_string(index=False))
