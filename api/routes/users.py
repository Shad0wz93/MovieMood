from contextlib import asynccontextmanager
from typing import List
import pandas as pd
from api.config import DATA_DIR
from fastapi import APIRouter, HTTPException, Request

# Variable globale pour stocker les données
ratings_df = None

@asynccontextmanager
async def lifespan(app):
    # Startup
    global ratings_df
    ratings_df = pd.read_csv(DATA_DIR / "ratings_small.csv")
    print(f"✅ {len(ratings_df)} ratings chargés")
    yield

router = APIRouter(
    prefix="",
    tags=["users"],
    lifespan=lifespan
)

@router.get("/users", response_model=List[int])
async def get_all_user_ids():
    """Récupère tous les IDs utilisateurs uniques"""
    user_ids = ratings_df['userId'].unique().tolist()
    return sorted(user_ids)

@router.get("/users/{user_id}/seen")
async def get_user_seen_movies(user_id: int, request: Request):
    """
    Retourne la liste des films déjà vus par l'utilisateur
    avec détails minimum : movieId, title et genre si disponible
    """
    rec_service = request.app.state.rec_service
    if rec_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    seen_movies_ids = rec_service.get_seen_movies(user_id)
    if seen_movies_ids is None:
        raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} non trouvé")

    # Récupère les infos sur les films vus
    seen_movies_details = rec_service.get_seen_movies(user_id)

    return {
        "user_id": user_id,
        "seen_count": len(seen_movies_details),
        "seen_movies": seen_movies_details[:50]  # Limite à 50 films pour la réponse
    }


@router.get("/users/{user_id}/stats")
async def get_user_stats(user_id: int, request: Request):
    """
    Retourne les statistiques détaillées d'un utilisateur
    """
    rec_service = request.app.state.rec_service
    if rec_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    stats = rec_service.get_user_stats(user_id)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Utilisateur {user_id} non trouvé")

    return stats