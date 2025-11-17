from fastapi import FastAPI
from api.routes import movies

app = FastAPI(title="Movies API")

app.include_router(movies.router)

@app.get("/")
async def root():
    return {"message": "Movies API"}