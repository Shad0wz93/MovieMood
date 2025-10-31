import pandas as pd
from scipy.sparse import coo_matrix

def pre_processing(movies,ratings, user_id):
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

    # Tranforme les user et movies en id uniques
    user_ids = ratings["userId"].astype('category')
    item_ids = ratings["movieId"].astype('category')

    # Créé une matrice pour stocker toutes notes
    # Chaque ligne correspond à un utilisateur
    # Chaque colonne correspond à un film
    matrix = coo_matrix((ratings['rating'], (user_ids.cat.codes, item_ids.cat.codes)))

    # Convertit la matrice en csr (compressed sparse row) pour faciliter la recherche (et rend compatible la fonction utilisée après)
    matrix_csr = matrix.tocsr()

    # On récupère l'index de cet utilisateur dans la liste de tous les user_ids
    user_index = user_ids.cat.categories.get_loc(user_id)


    return matrix_csr, user_index, item_ids