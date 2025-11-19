from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import movies, ratings

app = FastAPI(title="Movies API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

app.include_router(movies.router)
app.include_router(ratings.router)


@app.get("/")
async def root():
    return {"message": "Movies API"}