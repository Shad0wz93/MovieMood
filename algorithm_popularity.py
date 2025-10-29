import pandas
import pandas as pd

df_movies = pd.read_csv("tmdb_5000_movies.csv")
df_ratings = pd.read_csv("ratings_small.csv")

C= df_movies['vote_average'].mean()
print(C)

m = df_movies['vote_count'].quantile(0.9)
print(m)

q_movies: pandas.DataFrame = df_movies.copy().loc[df_movies['vote_count'] >= m]
print(q_movies.shape)

def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies["score"] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[["title", "vote_count", "vote_average", "score"]].head(10))