import pandas as pd
from src.data_preprocessing import load_and_process_movies
from src.content_based import build_content_model
from src.collaborative import build_als_model
from src.hybrid_adaptive import get_adaptive_recommendations
from scipy.sparse import coo_matrix

# -------------------------------
# 1️⃣ Charger les données
# -------------------------------
movies = load_and_process_movies(
    'data/tmdb_5000_movies.csv',
    'data/tmdb_5000_credits.csv'
)

ratings_file = 'data/ratings_small.csv'
ratings = pd.read_csv(ratings_file)[['userId','movieId','rating','timestamp']]

# -------------------------------
# 2️⃣ Construire les modèles
# -------------------------------
cosine_sim, indices = build_content_model(movies)
als_model, ratings = build_als_model(ratings)

# Créer les catégories pour ALS
ratings['userId_cat'] = ratings['userId'].astype('category')
ratings['movieId_cat'] = ratings['movieId'].astype('category')
ratings_csr = coo_matrix(
    (ratings['rating'],
     (ratings['userId_cat'].cat.codes, ratings['movieId_cat'].cat.codes))
).tocsr()

# -------------------------------
# 3️⃣ Utilisateur interactif
# -------------------------------
user_id = 2

while True:
    # Générer recommandations adaptatives
    recommendations = get_adaptive_recommendations(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        indices=indices,
        cosine_sim=cosine_sim,
        model=als_model,
        ratings_csr=ratings_csr,
        top_n=10
    )

    if recommendations.empty:
        print("Aucune recommandation disponible.")
        break

    print("\n🎬 Films proposés :")
    for idx, row in recommendations.head(5).iterrows():
        print(f"{idx+1}. {row['title']} (note moyenne: {row['vote_average']:.1f}, votes: {row['vote_count']})")

    # Choix utilisateur
    try:
        choice = int(input("\nEntrez le numéro du film que vous voulez noter (0 pour quitter) : "))
    except ValueError:
        print("Entrée invalide.")
        continue

    if choice == 0:
        break
    if choice < 1 or choice > len(recommendations):
        print("Numéro invalide.")
        continue

    selected_movie = recommendations.iloc[choice-1]
    vote = input(f"Aimez-vous ce film ? (y=like / n=dislike) : ").strip().lower()
    rating_value = 5 if vote == 'y' else 2

    # Ajouter le feedback dans ratings (en mémoire uniquement)
    feedback_row = {
        'userId': user_id,
        'movieId': selected_movie['id'],
        'rating': rating_value,
        'timestamp': pd.Timestamp.now()
    }
    ratings = pd.concat([ratings, pd.DataFrame([feedback_row])], ignore_index=True)

    # Mise à jour vote_count et vote_average (en mémoire)
    mask = movies['id'] == selected_movie['id']
    old_count = movies.loc[mask, 'vote_count'].values[0]
    old_avg   = movies.loc[mask, 'vote_average'].values[0]
    new_count = old_count + 1
    new_avg   = (old_avg * old_count + rating_value) / new_count
    movies.loc[mask, 'vote_count'] = new_count
    movies.loc[mask, 'vote_average'] = new_avg

    print(f"Votre note pour {selected_movie['title']} a été enregistrée !\n")

print("\nMerci ! Vos recommandations adaptatives sont terminées.")