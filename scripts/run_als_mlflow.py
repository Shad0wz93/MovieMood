import pandas as pd
import os
import mlflow
import mlflow.pyfunc

from utils.pre_processing import pre_processing
from utils.processing import processing
from utils.post_processing import post_processing
from als_model_wrapper import ALSModelWrapper   # <-- IMPORTANT

# ---------------------------
# Configuration chemins
# ---------------------------
movies_path = 'tmdb_5000_movies.csv'
ratings_path = 'ratings_small.csv'
credits_path = 'tmdb_5000_credits.csv'  # non utilisé mais vérifié

for file_path in [movies_path, ratings_path, credits_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier manquant : {file_path}")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
credits = pd.read_csv(credits_path)

# Création du dossier outputs si nécessaire
os.makedirs('outputs', exist_ok=True)

# Choix de l'utilisateur
user_id = int(input("Entrez l'ID de l'utilisateur pour les recommandations : "))

# ---------------------------
# Configuration MLflow
# ---------------------------
mlflow.set_tracking_uri("http://localhost:5001")  # ton serveur MLflow
mlflow.set_experiment("ALS_User_Recommendations")

with mlflow.start_run(run_name=f"ALS_User_{user_id}"):

    # Log paramètre utilisateur
    mlflow.log_param("user_id", user_id)

    # Pipeline ALS
    matrix_csr, user_index, item_ids = pre_processing(movies, ratings, user_id)
    ids, scores = processing(matrix_csr, user_index)
    recommended_df, user_ratings_sorted = post_processing(movies, ratings, item_ids, ids, scores, user_id)

    # ----------------------------------------------------
    # Nouveau : Logging du modèle dans MLflow (onglet Models)
    # ----------------------------------------------------
    model_path = "als_recommendation_model"

    mlflow.pyfunc.log_model(
        artifact_path=model_path,
        python_model=ALSModelWrapper(ids, scores)
    )

    print(f"📦 Modèle ALS enregistré dans MLflow --> {model_path}")

    # Sauvegarde CSV + log artifact
    output_file = f'outputs/recommended_movies_{user_id}.csv'
    recommended_df.to_csv(output_file, index=False)
    mlflow.log_artifact(output_file)

# ---------------------------
# Affichage console
# ---------------------------
print(f"\n✅ Recommandations sauvegardées dans : {output_file}\n")

print(f"🎬 Films notés par l'utilisateur {user_id} :\n")
print(user_ratings_sorted.head(10).to_string(index=False))

print(f"\n⭐ Top recommandations pour l'utilisateur {user_id} :\n")
print(recommended_df.head(10).to_string(index=False))

print(f"\n🏃 View run ALS_User_{user_id} at: http://localhost:5001/#/experiments/0/runs\n")
