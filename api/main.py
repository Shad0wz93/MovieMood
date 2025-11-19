from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import recommendations, users
from api.services.recommendation_service import RecommendationService
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle hybride au démarrage"""
    print("🚀 Chargement du modèle hybride...")
    rec_service = RecommendationService()
    app.state.rec_service = rec_service
    print("✅ Modèle chargé avec succès")
    yield
    print("🔻 Arrêt du serveur")


app = FastAPI(
    title="MovieMood API",
    description="API de recommandation de films avec modèle hybride (Logistique + SVD)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(recommendations.router, prefix="/api/v1", tags=["recommendations"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])


@app.get("/")
async def root():
    return {
        "message": "🎬 MovieMood API - Recommandations hybrides",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/predict": "Obtenir des recommandations personnalisées",
            "POST /api/v1/explain": "Expliquer pourquoi ces films ont été recommandés",
            "GET /api/v1/users/{user_id}/seen": "Films déjà vus par l'utilisateur",
            "GET /api/v1/users/{user_id}/stats": "Statistiques de l'utilisateur",
            "GET /health": "Vérifier l'état du service"
        },
        "docs": "/docs",
        "description": "Utilisez /predict pour obtenir des recommandations, puis /explain pour comprendre pourquoi"
    }


@app.get("/health")
async def health_check():
    if not hasattr(app.state, 'rec_service') or app.state.rec_service is None:
        return {"status": "unhealthy", "message": "Service non initialisé"}

    return {
        "status": "healthy",
        "model": "hybrid (logistic + SVD)",
        "users_count": len(app.state.rec_service.seen_dict),
        "movies_count": len(app.state.rec_service.movies),
        "threshold": app.state.rec_service.threshold
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)