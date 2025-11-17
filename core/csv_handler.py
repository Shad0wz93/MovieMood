import pandas as pd
import json
from pathlib import Path
from typing import List
from models.movie import Movie


class CSVHandler:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> List[Movie]:
        self.df = pd.read_csv(self.csv_path)
        movies = []

        for _, row in self.df.iterrows():
            # Parse le champ genres depuis JSON string vers liste
            genres = json.loads(row['genres']) if pd.notna(row['genres']) else []
            genre_names = [g['name'] for g in genres]

            movies.append(Movie(
                title=row['title'],
                release_date=row['release_date'],
                imdb_id=row['imdb_id'],
                genres=genre_names,
                runtime=int(row['runtime']) if pd.notna(row['runtime']) else 0
            ))

        return movies