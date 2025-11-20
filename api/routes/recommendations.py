from fastapi import APIRouter, HTTPException, Request

from api.models import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_recommendations(request_body: PredictRequest, request: Request):
    """
    🎯 Route /predict : Retourne les recommandations top-K

    Cette route retourne simplement les N meilleurs films recommandés
    pour un utilisateur, avec leurs scores hybrides.
    """
    rec_service = request.app.state.rec_service
    if rec_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    recommendations = rec_service.predict(
        user_id=request_body.user_id,
        topk=request_body.n_recommendations,
        min_score=request_body.min_score,
        alpha=request_body.alpha
    )

    if recommendations is None:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {request_body.user_id} non trouvé ou aucune recommandation disponible"
        )

    return PredictResponse(
        user_id=request_body.user_id,
        recommendations=recommendations.to_dict('records'),
        model="hybrid",
        params={
            "alpha": request_body.alpha,
            "min_score": request_body.min_score,
            "threshold": rec_service.threshold
        }
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain_recommendations(request_body: ExplainRequest, request: Request):
    """
    💡 Route /explain : Explique POURQUOI ces films ont été recommandés

    Cette route fournit une explication détaillée pour une liste de films :
    - Scores de chaque modèle (logistique, SVD)
    - Score hybride et poids utilisés
    - Interprétation textuelle
    - Contexte utilisateur (moyenne, nombre de films vus, etc.)
    """
    rec_service = request.app.state.rec_service
    if rec_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    if not request_body.movie_ids or len(request_body.movie_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="La liste movie_ids ne peut pas être vide"
        )

    explanation = rec_service.explain(
        user_id=request_body.user_id,
        movie_ids=request_body.movie_ids,
        alpha=request_body.alpha
    )

    if explanation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {request_body.user_id} non trouvé"
        )

    return ExplainResponse(**explanation)