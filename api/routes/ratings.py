from fastapi import APIRouter
from typing import List
from contextlib import asynccontextmanager
import pandas as pd
from config import DATA_DIR

# Variable globale pour stocker les données
ratings_df = None

@asynccontextmanager
async def lifespan(app):
    # Startup
    global ratings_df
    ratings_df = pd.read_csv(DATA_DIR / "ratings_small.csv")
    print(f"✅ {len(ratings_df)} ratings chargés")
    yield
    # Shutdown (si besoin de cleanup)

router = APIRouter(
    prefix="/ratings",
    tags=["ratings"],
    lifespan=lifespan
)

@router.get("/users", response_model=List[int])
async def get_all_user_ids():
    """Récupère tous les IDs utilisateurs uniques"""
    user_ids = ratings_df['userId'].unique().tolist()
    return sorted(user_ids)