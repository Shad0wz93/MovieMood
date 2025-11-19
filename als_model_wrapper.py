import mlflow.pyfunc
import pandas as pd

class ALSModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, user_index, item_ids):
        """
        model : ALS entraîné (AlternatingLeastSquares)
        user_index : index de l'utilisateur cible
        item_ids : Series des movieIds
        """
        self.model = model
        self.user_index = user_index
        self.item_ids = item_ids

    def predict(self, context, model_input):
        """
        model_input : DataFrame contenant 'user_id' (optionnel)
        Retourne DataFrame des recommandations
        """
        ids, scores = self.model.recommend(self.user_index, self.model.user_items[self.user_index])
        movie_labels = dict(enumerate(self.item_ids.cat.categories))
        recommended_movies = [(movie_labels[iid], float(score)) for iid, score in zip(ids, scores)]
        return pd.DataFrame(recommended_movies, columns=['movieId', 'score'])
