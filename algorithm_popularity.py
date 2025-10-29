import pandas
import pandas as pd

# Récupération du fichier des films
df_movies = pd.read_csv("tmdb_5000_movies.csv")

# Calcul de la note médiane des films
C= df_movies['vote_average'].mean()

# Récupération du nombre de votes qu'il faut pour être dans le top 10% des films avec le plus de votes
m = df_movies['vote_count'].quantile(0.9)

# Création d'une nouvelle dataframe avec seulement les 10% de films avec le plus de votes
q_movies: pandas.DataFrame = df_movies.copy().loc[df_movies['vote_count'] >= m]

# Fonction de calcul des scores de films basé sur le calcul de imdb
def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v+m) * R) + (m/(m+v) * C)


# On applique la fonction de calcul de poids à chaque ligne (axis = 1) du dataframe et on l'ajoute à une colonne 'score'
q_movies["score"] = q_movies.apply(weighted_rating, axis=1)

# On classe le dataframe par score descendant
q_movies = q_movies.sort_values('score', ascending=False)

# On affiche les 10 films avec le meilleur score
print(q_movies[["title", "vote_count", "vote_average", "score"]].head(10))