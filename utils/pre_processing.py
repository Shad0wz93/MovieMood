import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pre_processing(movies, ratings, user_id, n_components=20):
    # On s'assure que l'id des films est numérique
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

    # Transformation en catégories
    user_ids = ratings["userId"].astype('category')
    item_ids = ratings["movieId"].astype('category')

    # Extraction des notes
    rating_values = ratings['rating'].values.reshape(-1,1)

    # Application du scaler
    scaler = StandardScaler()
    rating_scaled = scaler.fit_transform(rating_values).flatten()

    # Création de la matrice utilisateur × film
    matrix = coo_matrix((rating_scaled, (user_ids.cat.codes, item_ids.cat.codes)))
    matrix_csr = matrix.tocsr()

    # Réduction de dimension via PCA
    pca = PCA(n_components=min(n_components, matrix_csr.shape[1]))
    matrix_reduced = pca.fit_transform(matrix_csr.toarray())
    matrix_csr = csr_matrix(matrix_reduced)

    # Récupération de l'index utilisateur
    user_index = user_ids.cat.categories.get_loc(user_id)

    return matrix_csr, user_index, item_ids
