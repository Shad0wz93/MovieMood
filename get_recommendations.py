def get_recommendations(title, df_movies, indices, cosine_sim):
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