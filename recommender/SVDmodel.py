#---- Partie 1 : chargement & nettoyage -------#

import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

import mlflow

#-- Chargement
df = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
dr = pd.read_csv("../data/ratings_small.csv", low_memory=False)

# Nettoyage movies
movies = df[["id", "title"]].copy()
movies["movieId"] = pd.to_numeric(movies["id"], errors="coerce")
movies = movies.dropna(subset=["movieId"])
movies["movieId"] = movies["movieId"].astype(int)
movies = movies[["movieId", "title"]]

print(movies.head())

# Nettoyage ratings
dr = dr[["userId", "movieId", "rating", "timestamp"]].copy()
dr = dr.dropna()
dr["userId"] = dr["userId"].astype(int)
dr["movieId"] = dr["movieId"].astype(int)
dr["rating"] = dr["rating"].astype(float)
dr = dr.sort_values("timestamp")
dr = dr.drop_duplicates(subset=["userId", "movieId"], keep="last")

#---- Partie 2 : dataset surprise + train/test -------#

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dr[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.1, random_state=42)

#---- Partie 3 : entraînement SVD + MLflow -------#

mlflow.set_experiment("MovieMood-Reco")

with mlflow.start_run(run_name="SVD_v2"):

    algo = SVD(
        n_factors=50,
        n_epochs=20,
        random_state=42
    )

    print("🚂 Entraînement SVD...")
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    print(f"📏 RMSE = {rmse:.4f}")

    mlflow.log_param("model_type", "SVD")
    mlflow.log_param("n_factors", 50)
    mlflow.log_param("n_epochs", 20)
    mlflow.log_metric("rmse", rmse)

#---- Partie 4 : recomm
