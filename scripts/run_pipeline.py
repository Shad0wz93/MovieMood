import pandas as pd
import os
import sys

# Ajout du dossier parent pour reconnaître utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pre_processing import pre_processing
from utils.processing import processing
from utils.post_processing import post_processing

# Configuration
movies_path = 'tmdb_5000_movies.csv'
ratings_path = 'ratings_small.csv'
credits_path = 'tmdb_5000_credits.csv'  # pour info, non utilisé ici

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

# Pipeline ALS
matrix_csr, user_index, item_ids = pre_processing(movies, ratings, user_id)
ids, scores = processing(matrix_csr, user_index)
recommended_df, user_ratings_sorted = post_processing(movies, ratings, item_ids, ids, scores, user_id)

# Sauvegarde
output_file = f'outputs/recommended_movies_{user_id}.csv'
recommended_df.to_csv(output_file, index=False)
print(f"\n✅ Recommandations sauvegardées dans : {output_file}\n")

# Affichage console
print(f"🎬 Films notés par l'utilisateur {user_id} :\n")
print(user_ratings_sorted.head(10).to_string(index=False))

print(f"\n⭐ Top recommandations pour l'utilisateur {user_id} :\n")
print(recommended_df.head(10).to_string(index=False))
