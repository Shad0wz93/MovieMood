from get_recommendations import get_recommendations

import pandas as pd

# On importe les fonctions nécessaires depuis sklearn
# TfIdfVectorizer pour faire de la magie
from sklearn.feature_extraction.text import TfidfVectorizer

# Linear kernel pour faire d'autre magie
from sklearn.metrics.pairwise import linear_kernel

# Récupération du fichier des films
df_movies = pd.read_csv("tmdb_5000_movies.csv")

# Remplace les overview null par string vide
df_movies["overview"] = df_movies["overview"].fillna("")

# Initialisation du vectorizer; Suppression des "mots vides" (comme "a", "the"...)
tfidf = TfidfVectorizer(stop_words='english')

# Transforme les overviews en vecteur pour compréhension et analyse par la machine
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

# Compare la similarité entre toutes les overviews de films
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Mapping des films par leur titre
indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()


print(get_recommendations('The Avengers', df_movies, indices, cosine_sim))