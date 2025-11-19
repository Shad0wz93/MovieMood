import pandas as pd
import json
import ast
from pathlib import Path
from typing import List
from models.movie import Movie


class CSVHandler:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> List[Movie]:
        # Ajoute low_memory=False pour éviter le warning
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        movies = []

        for _, row in self.df.iterrows():
            try:
                # Essaye d'abord avec json.loads
                if pd.notna(row['genres']) and row['genres']:
                    try:
                        genres = json.loads(row['genres'])
                    except json.JSONDecodeError:
                        # Si ça échoue, essaye avec ast.literal_eval (pour les ' au lieu de ")
                        genres = ast.literal_eval(row['genres'])

                    genre_names = [g['name'] for g in genres] if genres else []
                else:
                    genre_names = []

                # Vérifie que les champs requis existent
                if pd.notna(row['title']) and pd.notna(row['imdb_id']):
                    movies.append(Movie(
                        title=str(row['title']),
                        release_date=str(row['release_date']) if pd.notna(row['release_date']) else "",
                        imdb_id=str(row['imdb_id']),
                        genres=genre_names,
                        runtime=int(float(row['runtime'])) if pd.notna(row['runtime']) and row['runtime'] != '' else 0
                    ))
            except Exception as e:
                # Ignore les lignes problématiques
                print(f"⚠️  Ligne ignorée : {e}")
                continue

        return movies