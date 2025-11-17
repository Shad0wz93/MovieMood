import pandas as pd
from scipy.sparse import coo_matrix

def pre_processing(movies, ratings, user_id):
    # On s'assure que l'id des films est numérique
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

    # Transformation en catégories
    user_ids = ratings["userId"].astype('category')
    item_ids = ratings["movieId"].astype('category')

    # Création de la matrice utilisateur × film
    matrix = coo_matrix((ratings['rating'], (user_ids.cat.codes, item_ids.cat.codes)))
    matrix_csr = matrix.tocsr()

    # Récupération de l'index utilisateur
    user_index = user_ids.cat.categories.get_loc(user_id)

    return matrix_csr, user_index, item_ids
