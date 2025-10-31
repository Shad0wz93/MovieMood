import pandas as pd
from ast import literal_eval
from .utils import clean_data, create_soup, get_director, get_list


def load_and_process_movies(movies_file, credits_file):
    # Charger films et crédits
    movies = pd.read_csv(movies_file)[
        ['id', 'title', 'original_title', 'genres', 'keywords', 'overview', 'vote_average', 'vote_count', 'runtime']]
    credits = pd.read_csv(credits_file)[['movie_id', 'title', 'cast', 'crew']]

    # Renommer pour éviter doublon
    credits.columns = ['id', 'title_credit', 'cast', 'crew']

    # Merge
    movies = movies.merge(credits, on='id', how='left')

    # Convertir JSON string → objets Python
    for feature in ['cast', 'crew', 'genres', 'keywords']:
        movies[feature] = movies[feature].apply(literal_eval)

    # Extraire directeur et top 3 cast/genres/keywords
    movies['director'] = movies['crew'].apply(get_director)
    movies['cast'] = movies['cast'].apply(get_list)
    movies['genres'] = movies['genres'].apply(get_list)
    movies['keywords'] = movies['keywords'].apply(get_list)

    # Nettoyage
    for feature in ['cast', 'keywords', 'director', 'genres']:
        movies[feature] = movies[feature].apply(clean_data)

    # Créer la 'soup'
    movies['soup'] = movies.apply(create_soup, axis=1)

    # Supprimer les films sans titre
    movies = movies[movies['title'].notna()]

    return movies
