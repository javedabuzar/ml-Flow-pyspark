from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "flood data" / "train.csv"
MLRUNS_DIR = ROOT / "mlruns"
MODELS_DIR = ROOT / "models"


def mlflow_tracking_uri() -> str:
    """file:// URI required on Windows (bare D:\\ paths break MLflow)."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return MLRUNS_DIR.resolve().as_uri()
EXPERIMENT_NAME = "flood-risk-pyspark"
REGISTERED_MODEL_NAME = "flood-probability-regressor"

FEATURE_COLUMNS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

TARGET_COLUMN = "FloodProbability"
