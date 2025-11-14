import pandas as pd
from utils.pre_processing import pre_processing

def test_pre_processing_creates_sparse_matrix():
    movies = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["A", "B", "C"]
    })
    ratings = pd.DataFrame({
        "userId": [1, 2, 2],
        "movieId": [1, 2, 3],
        "rating": [4.0, 5.0, 3.0]
    })

    matrix_csr, user_index, item_ids = pre_processing(movies, ratings, user_id=2)

    # Vérifie les types et dimensions
    assert matrix_csr.shape[0] == ratings["userId"].nunique()
    assert matrix_csr.shape[1] == ratings["movieId"].nunique()
    assert isinstance(user_index, int)
    assert len(item_ids) == len(ratings)
