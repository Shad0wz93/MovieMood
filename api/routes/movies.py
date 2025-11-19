from fastapi import APIRouter
from typing import List
from contextlib import asynccontextmanager
from models.movie import Movie
from core.csv_handler import CSVHandler
from api.config import DATA_DIR

# Variables globales
csv_handler = CSVHandler(DATA_DIR / "movies_metadata.csv")
movies_data: List[Movie] = []

@asynccontextmanager
async def lifespan(app):
    # Startup
    global movies_data
    movies_data = csv_handler.load_data()
    print(f"✅ {len(movies_data)} films chargés")
    yield
    # Shutdown (optionnel)

router = APIRouter(
    prefix="/movies",
    tags=["movies"],
    lifespan=lifespan
)

@router.get("/", response_model=List[Movie])
async def get_all_movies():
    """Récupère tous les films"""
    return movies_data