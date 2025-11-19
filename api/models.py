from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ============= PREDICT =============

class PredictRequest(BaseModel):
    """Requête pour obtenir des recommandations"""
    user_id: int = Field(..., description="ID de l'utilisateur", example=2)
    n_recommendations: int = Field(10, ge=1, le=50, description="Nombre de recommandations souhaitées")
    min_score: float = Field(0.6, ge=0.0, le=1.0, description="Score hybride minimum (0-1)")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Poids du modèle logistique dans l'hybride (0=SVD seul, 1=Logit seul)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 2,
                "n_recommendations": 10,
                "min_score": 0.6,
                "alpha": 0.5
            }
        }


class PredictResponse(BaseModel):
    """Réponse avec les recommandations"""
    user_id: int
    recommendations: List[Dict[str, Any]] = Field(..., description="Liste des films recommandés avec leurs scores")
    model: str = Field(..., description="Type de modèle utilisé")
    params: Dict[str, float] = Field(..., description="Paramètres utilisés pour la prédiction")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 2,
                "recommendations": [
                    {
                        "movieId": 123,
                        "title": "The Matrix",
                        "hybrid_score": 0.85
                    },
                    {
                        "movieId": 456,
                        "title": "Inception",
                        "hybrid_score": 0.82
                    }
                ],
                "model": "hybrid",
                "params": {
                    "alpha": 0.5,
                    "min_score": 0.6,
                    "threshold": 4.0
                }
            }
        }


# ============= EXPLAIN =============

class ExplainRequest(BaseModel):
    """Requête pour expliquer des recommandations"""
    user_id: int = Field(..., description="ID de l'utilisateur", example=2)
    movie_ids: List[int] = Field(..., description="Liste des IDs de films à expliquer", example=[123, 456, 789])
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Poids du modèle logistique (doit correspondre à celui utilisé pour /predict)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 2,
                "movie_ids": [123, 456, 789],
                "alpha": 0.5
            }
        }


class MovieExplanation(BaseModel):
    """Explication détaillée pour un film"""
    movieId: int
    title: str
    scores: Dict[str, float] = Field(..., description="Scores de chaque modèle")
    weights: Dict[str, float] = Field(..., description="Poids utilisés dans l'hybride")
    interpretation: str = Field(..., description="Interprétation textuelle des scores")
    already_seen: bool = Field(..., description="Si l'utilisateur a déjà vu ce film")
    user_rating: Optional[float] = Field(None, description="Note donnée par l'utilisateur si déjà vu")


class UserContext(BaseModel):
    """Contexte utilisateur pour l'explication"""
    total_ratings: int
    average_rating: float
    seen_movies: int


class ExplainResponse(BaseModel):
    """Réponse avec les explications détaillées"""
    user_id: int
    user_context: UserContext
    model_params: Dict[str, float]
    explanations: List[MovieExplanation]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 2,
                "user_context": {
                    "total_ratings": 150,
                    "average_rating": 3.8,
                    "seen_movies": 150
                },
                "model_params": {
                    "alpha": 0.5,
                    "threshold": 4.0
                },
                "explanations": [
                    {
                        "movieId": 123,
                        "title": "The Matrix",
                        "scores": {
                            "logistic": 0.82,
                            "svd": 0.88,
                            "hybrid": 0.85
                        },
                        "weights": {
                            "logistic": 0.5,
                            "svd": 0.5
                        },
                        "interpretation": "🔥 Fortement recommandé : forte probabilité d'appréciation selon vos goûts passés et note prédite élevée par filtrage collaboratif",
                        "already_seen": False,
                        "user_rating": None
                    }
                ]
            }
        }