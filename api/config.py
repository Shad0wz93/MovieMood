#from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data"


"""class Settings(BaseSettings):
    app_name: str = "MovieMood API"
    app_version: str = "1.0.0"
    data_path: str = "../data/"
    threshold: float = 4.0
    default_alpha: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()"""