import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    DISCLAIMER,
    FEATURE_ORDER,
    MODEL_NAME,
    THRESHOLD,
    ModelService,
    get_model_service,
)
from .schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)


def configured_origins() -> list[str]:
    configured = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Educational model-risk estimation service.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model service unavailable") from exc
    return HealthResponse(status="ready", model_ready=True)


@app.get("/metadata", response_model=MetadataResponse)
def metadata(
    service: ModelService = Depends(get_model_service),
) -> MetadataResponse:
    del service
    return MetadataResponse(
        model=MODEL_NAME,
        feature_count=len(FEATURE_ORDER),
        threshold=THRESHOLD,
        educational_only=True,
        disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    result = service.predict(payload)
    return PredictionResponse(
        probability=result.probability,
        percentage=result.percentage,
        interpretation=result.interpretation,
        threshold=THRESHOLD,
        model=MODEL_NAME,
        feature_count=len(FEATURE_ORDER),
    )
