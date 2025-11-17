from pydantic import BaseModel
from typing import List

class Movie(BaseModel):
    title: str
    release_date: str
    imdb_id: str
    genres: List[str]
    runtime: int