import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1) Charger les données
df_movies = pd.read_csv("tmdb_5000_movies.csv")
df_movies['overview'] = df_movies['overview'].fillna('')

# 2) TF-IDF sur overview
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

# 3) Matrice de similarité cosinus
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 4) Associer chaque titre à son index
indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

# 5) Fonction de recommandation
def get_recommendations(title, topn=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]  # ignorer le film lui-même
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices]

# 6) Test dans le terminal
if __name__ == "__main__":
    movie = input("Entrez le nom d'un film : ")
    if movie in indices:
        print("\n Films recommandés :\n")
        recs = get_recommendations(movie, topn=10)
        for i, title in enumerate(recs, 1):
            print(f"{i}. {title}")
    else:
        print("\n Film non trouvé. Vérifiez l'orthographe.")
