from implicit.als import AlternatingLeastSquares

def processing(matrix_csr, user_index):
    # La magie
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15)
    model.fit(matrix_csr)

    # La magie
    ids, scores = model.recommend(user_index, matrix_csr[user_index])

    return ids, scores