from fastapi import APIRouter
from typing import List
from models.movie import Movie
from core.csv_handler import CSVHandler
from config import DATA_DIR

router = APIRouter(
    prefix="/movies",
    tags=["movies"]
)

csv_handler = CSVHandler(DATA_DIR / "movies.csv")
movies_data: List[Movie] = []

@router.on_event("startup")
async def load_movies():
    global movies_data
    movies_data = csv_handler.load_data()
    print(f"✅ {len(movies_data)} films chargés")

@router.get("/", response_model=List[Movie])
async def get_all_movies():
    """Récupère tous les films"""
    return movies_data