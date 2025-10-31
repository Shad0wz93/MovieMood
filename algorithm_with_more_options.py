# Transforme les strings en objets
from ast import literal_eval

import pandas as pd

import numpy as np

from get_recommendations import get_recommendations

# Récupération du fichier des films et des crédits
df_movies = pd.read_csv("tmdb_5000_movies.csv")
df_credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge les crédits dans la dataframe movie
df_credits.columns = ['id','tittle','cast','crew']
df_movies = df_movies.merge(df_credits,on='id')

# Transforme les string des valeurs voulues en objets
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(literal_eval)


# Renvoie le nom du réalisateur
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Renvoie les 3 premiers 'name' dans une list
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]

        if len(x) > 3:
            names = names[:3]

        return names

    return []


# On renvoie le réalisateur pour chaque film
df_movies['director'] = df_movies['crew'].apply(get_director)

# On renvoie les 3 premières valeurs pour les cast, mot clés et genres de films
features =  ['cast', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(get_list)

# Fonction pour clean les string
# Strip tous les espaces
# Met tout en lowercase
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Vérifie si le réalisateur existe (sinon renvoie string vide)
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Applique la fonction clean sur les valeurs de chaque film
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(clean_data)

# Joint tous les string des valeurs voulues (cast, mot clés, réalisateur, genres) dans un seul string pour chaque film
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df_movies['soup'] = df_movies.apply(create_soup, axis=1)


# On importe CountVectorizer au lieu de TfIdf
# Pour ne pas casser si un acteur / réalisateur apparaît dans plus de films
from sklearn.feature_extraction.text import CountVectorizer

# Initialisation du vectorizer; Suppression des "mots vides" (comme "a", "the"...)
count = CountVectorizer(stop_words='english')

# Transforme les overviews en vecteur pour compréhension et analyse par la machine
count_matrix = count.fit_transform(df_movies['soup'])

# Cosine similarity pour les similarités
from sklearn.metrics.pairwise import cosine_similarity

# Compare la similarité entre toutes les données de films
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Mapping des films par leur titre
indices = pd.Series(df_movies.index, index=df_movies['title'])

print(get_recommendations('The Avengers', df_movies, indices, cosine_sim2))