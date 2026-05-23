"""FastAPI service: PySpark MLflow model inference + frontend."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import FEATURE_COLUMNS, mlflow_tracking_uri
from src.prediction import load_model, predict_one
from src.spark_session import create_spark

STATIC_DIR = Path(__file__).resolve().parent / "static"
spark = None
model = None


class FloodInput(BaseModel):
    MonsoonIntensity: float = Field(5, ge=0, le=10)
    TopographyDrainage: float = Field(5, ge=0, le=10)
    RiverManagement: float = Field(5, ge=0, le=10)
    Deforestation: float = Field(5, ge=0, le=10)
    Urbanization: float = Field(5, ge=0, le=10)
    ClimateChange: float = Field(5, ge=0, le=10)
    DamsQuality: float = Field(5, ge=0, le=10)
    Siltation: float = Field(5, ge=0, le=10)
    AgriculturalPractices: float = Field(5, ge=0, le=10)
    Encroachments: float = Field(5, ge=0, le=10)
    IneffectiveDisasterPreparedness: float = Field(5, ge=0, le=10)
    DrainageSystems: float = Field(5, ge=0, le=10)
    CoastalVulnerability: float = Field(5, ge=0, le=10)
    Landslides: float = Field(5, ge=0, le=10)
    Watersheds: float = Field(5, ge=0, le=10)
    DeterioratingInfrastructure: float = Field(5, ge=0, le=10)
    PopulationScore: float = Field(5, ge=0, le=10)
    WetlandLoss: float = Field(5, ge=0, le=10)
    InadequatePlanning: float = Field(5, ge=0, le=10)
    PoliticalFactors: float = Field(5, ge=0, le=10)


class PredictResponse(BaseModel):
    flood_probability: float
    risk_level: str
    model_source: str


def _risk_level(probability: float) -> str:
    if probability < 0.45:
        return "Low"
    if probability < 0.55:
        return "Medium"
    return "High"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global spark, model
    # Respect existing env var (e.g. Docker sets bare path); only fall back to
    # the file:// URI helper when running locally without the env set.
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri()
    spark = create_spark("flood-api")
    model = load_model(spark)
    yield
    if spark is not None:
        spark.stop()


app = FastAPI(
    title="Flood Risk Predictor",
    description="PySpark + MLflow flood probability API",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "API is running. POST /predict with feature JSON."}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features": len(FEATURE_COLUMNS),
    }


@app.get("/features")
async def features():
    return {"features": FEATURE_COLUMNS}


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: FloodInput):
    if model is None or spark is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = payload.model_dump()
        probability = predict_one(model, spark, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    source = os.getenv("MLFLOW_MODEL_URI") or "local/mlruns or models/spark_pipeline"
    return PredictResponse(
        flood_probability=round(probability, 4),
        risk_level=_risk_level(probability),
        model_source=source,
    )
