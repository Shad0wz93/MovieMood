import pandas
import pandas as pd

# On importe les fonctions nécessaires depuis sklearn
# TfIdfVectorizer pour faire de la magie
from sklearn.feature_extraction.text import TfidfVectorizer

# Linear kernel pour faire d'autre magie
from sklearn.metrics.pairwise import linear_kernel

# Récupération du fichier des films
df_movies = pd.read_csv("tmdb_5000_movies.csv")

print(df_movies.loc[[3144]]['overview'])

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


def get_recommendations(title, cosine_sim=cosine_sim):
    # On récupère l'index du film par son titre
    idx = indices[title]

    # Récupère les similarités de ce film avec tous les autres films
    sim_scores = list(enumerate(cosine_sim[idx]))

    # On classe les similarités par score descendant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # On récupère les 10 films les plus similaires
    sim_scores = sim_scores[1:11]

    # On récupère les index des films
    movie_indices = [i[0] for i in sim_scores]

    # On récupère les titres des films par leur index
    return df_movies['title'].iloc[movie_indices]


print(get_recommendations('The Avengers'))