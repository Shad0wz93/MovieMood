# recommendation.py
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ---------- 1) Charger & fusionner ----------
df_movies = pd.read_csv("tmdb_5000_movies.csv")
df_credits = pd.read_csv("tmdb_5000_credits.csv")

# le fichier credits a 'movie_id' -> on renomme pour fusionner
if "movie_id" in df_credits.columns:
    df_credits = df_credits.rename(columns={"movie_id": "id"})
# attention à la faute 'tittle' -> on garde 'title' de df_movies de toute façon
df_movies = df_movies.merge(df_credits[["id", "cast", "crew"]], on="id", how="left")
df_movies["overview"] = df_movies["overview"].fillna("")

# ---------- 2) Parser les colonnes JSON (string -> objets Python) ----------
for col in ["cast", "crew", "keywords", "genres"]:
    if col in df_movies.columns:
        df_movies[col] = df_movies[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

# ---------- 3) Extraction metadonnées utiles ----------
def get_directors(crew_list):
    return [m.get("name") for m in crew_list if isinstance(m, dict) and m.get("job") == "Director"]

def top_names(list_of_dicts, k=None):
    """Extrait la clé 'name' de chaque dict, limite à k si précisé."""
    if not isinstance(list_of_dicts, list):
        return []
    names = [d.get("name") for d in list_of_dicts if isinstance(d, dict) and d.get("name")]
    return names[:k] if k else names

# réalisateurs (liste), 3 acteurs, quelques keywords, tous les genres
df_movies["director"] = df_movies["crew"].apply(get_directors)
df_movies["cast"] = df_movies["cast"].apply(lambda x: top_names(x, k=3))
df_movies["keywords"] = df_movies["keywords"].apply(lambda x: top_names(x, k=10))
df_movies["genres"] = df_movies["genres"].apply(top_names)

# ---------- 4) Nettoyage (minuscule, sans espaces) ----------
def clean_list(lst):
    return [str(v).lower().replace(" ", "") for v in lst if v]

df_movies["director"] = df_movies["director"].apply(clean_list)
df_movies["cast"] = df_movies["cast"].apply(clean_list)
df_movies["keywords"] = df_movies["keywords"].apply(clean_list)
df_movies["genres"] = df_movies["genres"].apply(clean_list)

# ---------- 5) Créer la "soup" (poids réalisateur x3) ----------
def create_soup(row, director_weight=3):
    dir_tokens = row["director"] * director_weight if row["director"] else []
    tokens = row["keywords"] + row["cast"] + dir_tokens + row["genres"]
    return " ".join(tokens).strip()

df_movies["soup"] = df_movies.apply(create_soup, axis=1)

# ---------- 6) Vectorisation + similarité ----------
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df_movies["soup"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# remettre des index propres et construire l'index par titre
df_movies = df_movies.reset_index(drop=True)
indices = pd.Series(df_movies.index, index=df_movies["title"]).drop_duplicates()

# ---------- 7) Recommandation ----------
def get_recommendations(title, cosine_sim=cosine_sim2, topn=10):
    # tolérance faute de frappe
    if title not in indices.index:
        match = get_close_matches(title, indices.index.tolist(), n=1, cutoff=0.6)
        if not match:
            return []
        title = match[0]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : topn + 1]  # ignorer le film lui-même
    movie_indices = [i for i, _ in sim_scores]
    return df_movies["title"].iloc[movie_indices].tolist()

# ---------- 8) Test rapide + mode terminal ----------
if __name__ == "__main__":
    # test direct
    print(get_recommendations("The Dark Knight Rises")[:10])

    # interaction terminal
    try:
        movie = input("Entrez le nom d'un film : ").strip()
        recs = get_recommendations(movie, topn=10)
        if not recs:
            print("❌ Film non trouvé. Vérifiez l'orthographe.")
        else:
            print("\n🎬 Films recommandés :\n")
            for i, t in enumerate(recs, 1):
                print(f"{i}. {t}")
    except KeyboardInterrupt:
        pass
