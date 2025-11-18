import mlflow.pyfunc

class ALSModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ids, scores):
        self.ids = ids
        self.scores = scores

    def predict(self, context, model_input):
        """
        model_input : DataFrame (ex : user_id)
        retourne les ID de films recommandés.
        """
        return self.ids
