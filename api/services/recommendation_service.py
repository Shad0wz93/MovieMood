import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from surprise import Dataset, Reader, SVD
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RecommendationService:
    def __init__(self, data_path=None, threshold=4.0):
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data"
        self.threshold = threshold
        self.data_path = Path(data_path)
        self.load_data()
        self.train_models()

    def load_data(self):
        """Charge et nettoie les données"""
        print("📊 Chargement des données...")

        # Movies
        df = pd.read_csv(self.data_path / "movies_metadata.csv", low_memory=False)
        self.movies = df[["id", "title", "genres"]].copy()
        self.movies["movieId"] = pd.to_numeric(self.movies["id"], errors="coerce")
        self.movies = self.movies.dropna(subset=["movieId"])
        self.movies["movieId"] = self.movies["movieId"].astype(int)
        self.movies = self.movies[["movieId", "title", "genres"]]

        # Ratings
        dr = pd.read_csv(self.data_path / "ratings_small.csv", low_memory=False)
        dr = dr[["userId", "movieId", "rating", "timestamp"]].copy()
        dr["userId"] = pd.to_numeric(dr["userId"], errors="coerce")
        dr["movieId"] = pd.to_numeric(dr["movieId"], errors="coerce")
        dr["rating"] = pd.to_numeric(dr["rating"], errors="coerce")
        dr = dr.dropna(subset=["userId", "movieId", "rating"])

        dr = dr.sort_values("timestamp")
        dr = dr.drop_duplicates(subset=["userId", "movieId"], keep="last")
        dr["label"] = (dr["rating"] >= self.threshold).astype(int)

        self.ratings = dr

        # Films vus par utilisateur
        self.seen_dict = {}
        for _, row in dr.iterrows():
            user = int(row["userId"])
            movie = int(row["movieId"])
            if user not in self.seen_dict:
                self.seen_dict[user] = set()
            self.seen_dict[user].add(movie)

        print(f"✅ {len(self.movies)} films, {len(self.ratings)} évaluations, {len(self.seen_dict)} utilisateurs")

    def train_models(self):
        """Entraîne les modèles logistique et SVD"""
        print("🤖 Entraînement des modèles...")

        # Logistique
        self.encoder = OneHotEncoder(handle_unknown="ignore", dtype=float)
        X = self.encoder.fit_transform(self.ratings[["userId", "movieId"]])
        y = self.ratings["label"].to_numpy()

        self.logit = LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced")
        self.logit.fit(X, y)

        # SVD
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        self.svd = SVD(n_factors=50, n_epochs=20, random_state=42)
        self.svd.fit(trainset)

        print("✅ Modèles entraînés")

    def svd_score_normalized(self, user_id: int, movie_id: int) -> float:
        """Retourne le score SVD normalisé entre 0 et 1"""
        est = self.svd.predict(int(user_id), int(movie_id)).est
        score = (est - 0.5) / (5.0 - 0.5)
        return float(np.clip(score, 0.0, 1.0))

    def _get_candidates(self, user_id: int) -> Optional[np.ndarray]:
        """Retourne les films candidats (non vus) pour un utilisateur"""
        if user_id not in self.ratings["userId"].unique():
            return None

        seen_movies = self.seen_dict.get(user_id, set())
        all_movies = self.movies["movieId"].unique()
        candidates = [m for m in all_movies if m not in seen_movies]

        if not candidates:
            return None

        return np.array(candidates, dtype=int)

    def compute_detailed_scores(
            self,
            user_id: int,
            movie_ids: Optional[List[int]] = None,
            alpha: float = 0.5
    ) -> Optional[pd.DataFrame]:
        """
        Calcule les scores détaillés (logit, SVD, hybride) pour des films spécifiques
        ou pour tous les candidats si movie_ids est None.

        Retourne un DataFrame avec: movieId, title, logit_score, svd_score, hybrid_score
        """
        user_id = int(user_id)

        if movie_ids is None:
            candidates = self._get_candidates(user_id)
            if candidates is None:
                return None
        else:
            candidates = np.array([int(m) for m in movie_ids], dtype=int)

        # Scores logistiques
        cand_df = pd.DataFrame({
            "userId": [user_id] * len(candidates),
            "movieId": candidates
        })
        X_cand = self.encoder.transform(cand_df[["userId", "movieId"]])
        probs_log = self.logit.predict_proba(X_cand)[:, 1]

        # Scores SVD normalisés
        svd_scores = np.array([
            self.svd_score_normalized(user_id, m) for m in candidates
        ])

        # Scores hybrides
        hybrid_scores = alpha * probs_log + (1 - alpha) * svd_scores

        # Construction du DataFrame résultat
        result = pd.DataFrame({
            "movieId": candidates,
            "logit_score": probs_log,
            "svd_score": svd_scores,
            "hybrid_score": hybrid_scores
        })

        # Jointure avec les titres et genres
        result = result.merge(self.movies, on="movieId", how="left")

        return result

    def predict(
            self,
            user_id: int,
            topk: int = 10,
            min_score: float = 0.6,
            alpha: float = 0.5
    ) -> Optional[pd.DataFrame]:
        """Retourne les top-K recommandations"""
        user_id = int(user_id)
        detailed_scores = self.compute_detailed_scores(user_id, alpha=alpha)
        if detailed_scores is None:
            return None

        filtered = detailed_scores[detailed_scores["hybrid_score"] >= min_score]
        if len(filtered) == 0:
            return None

        top_recommendations = filtered.nlargest(topk, "hybrid_score")
        return top_recommendations[["movieId", "title", "hybrid_score"]].reset_index(drop=True)

    def explain(
            self,
            user_id: int,
            movie_ids: List[int],
            alpha: float = 0.5
    ) -> Optional[Dict]:
        """Explique pourquoi ces films ont été recommandés"""
        user_id = int(user_id)
        if user_id not in self.ratings["userId"].unique():
            return None

        detailed_scores = self.compute_detailed_scores(user_id, movie_ids, alpha)
        if detailed_scores is None or len(detailed_scores) == 0:
            return None

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        avg_rating = float(user_ratings["rating"].mean())
        total_ratings = len(user_ratings)

        explanations = []
        for _, row in detailed_scores.iterrows():
            movie_id = int(row["movieId"])
            already_rated = movie_id in self.seen_dict.get(user_id, set())
            user_rating = None
            if already_rated:
                rating_row = user_ratings[user_ratings["movieId"] == movie_id]
                if len(rating_row) > 0:
                    user_rating = float(rating_row.iloc[0]["rating"])

            explanation = {
                "movieId": movie_id,
                "title": row["title"],
                "scores": {
                    "logistic": round(float(row["logit_score"]), 4),
                    "svd": round(float(row["svd_score"]), 4),
                    "hybrid": round(float(row["hybrid_score"]), 4)
                },
                "weights": {
                    "logistic": alpha,
                    "svd": 1 - alpha
                },
                "interpretation": self._interpret_scores(
                    row["logit_score"],
                    row["svd_score"],
                    row["hybrid_score"]
                ),
                "already_seen": already_rated,
                "user_rating": user_rating
            }
            explanations.append(explanation)

        return {
            "user_id": user_id,
            "user_context": {
                "total_ratings": total_ratings,
                "average_rating": round(avg_rating, 2),
                "seen_movies": len(self.seen_dict.get(user_id, set()))
            },
            "model_params": {
                "alpha": alpha,
                "threshold": self.threshold
            },
            "explanations": explanations
        }

    def _interpret_scores(self, logit: float, svd: float, hybrid: float) -> str:
        """Génère une interprétation textuelle des scores"""
        interpretations = []
        if logit > 0.7:
            interpretations.append("forte probabilité d'appréciation selon vos goûts passés")
        elif logit > 0.5:
            interpretations.append("probabilité modérée d'appréciation")
        else:
            interpretations.append("probabilité faible selon le modèle de classification")

        if svd > 0.7:
            interpretations.append("note prédite élevée par filtrage collaboratif")
        elif svd > 0.5:
            interpretations.append("note prédite moyenne")
        else:
            interpretations.append("note prédite basse")

        if hybrid > 0.8:
            return "🔥 Fortement recommandé : " + " et ".join(interpretations)
        elif hybrid > 0.6:
            return "✅ Recommandé : " + " et ".join(interpretations)
        else:
            return "⚠️ Recommandation modérée : " + " et ".join(interpretations)

    def get_seen_movies(self, user_id: int) -> Optional[List[Dict]]:
        """Retourne la liste des films vus avec ID, titre et genre"""
        seen = self.seen_dict.get(int(user_id))
        if not seen:
            return None
        movies_seen = self.movies[self.movies["movieId"].isin(seen)]
        return movies_seen[["movieId", "title", "genres"]].to_dict(orient="records")

    def get_user_stats(self, user_id: int) -> Optional[Dict]:
        """Retourne les statistiques d'un utilisateur"""
        user_id = int(user_id)
        if user_id not in self.ratings["userId"].unique():
            return None
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        seen_movies = self.seen_dict.get(user_id, set())
        return {
            "user_id": user_id,
            "total_ratings": len(user_ratings),
            "average_rating": float(user_ratings["rating"].mean()),
            "seen_movies": len(seen_movies),
            "liked_movies": int((user_ratings["rating"] >= self.threshold).sum()),
            "rating_distribution": user_ratings["rating"].value_counts().sort_index().to_dict()
        }