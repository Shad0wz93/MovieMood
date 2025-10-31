import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix


def build_als_model(ratings):
    ratings['userId_cat'] = ratings['userId'].astype('category')
    ratings['movieId_cat'] = ratings['movieId'].astype('category')

    matrix = coo_matrix((ratings['rating'],
                         (ratings['userId_cat'].cat.codes, ratings['movieId_cat'].cat.codes)))
    matrix_csr = matrix.tocsr()

    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15)
    model.fit(matrix_csr)

    return model, ratings
