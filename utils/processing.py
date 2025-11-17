from implicit.als import AlternatingLeastSquares

def processing(matrix_csr, user_index):
    # Modèle ALS
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15)
    model.fit(matrix_csr)

    # Recommandations pour l'utilisateur
    ids, scores = model.recommend(user_index, matrix_csr[user_index])

    return ids, scores
