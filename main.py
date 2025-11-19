from fastapi import FastAPI
from api.routes import movies, ratings

app = FastAPI(title="Movies API")

app.include_router(movies.router)
app.include_router(ratings.router)

@app.get("/")
async def root():
    return {"message": "Movies API"}