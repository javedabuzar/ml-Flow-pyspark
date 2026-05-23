"""
API and PySpark pipeline tests — no pandas dependency.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Train a small model before running API tests."""
    env = os.environ.copy()
    env["TRAIN_MAX_ROWS"] = "3000"
    env["RF_NUM_TREES"] = "10"
    env["RF_MAX_DEPTH"] = "5"
    env["SPARK_SHUFFLE_PARTITIONS"] = "2"
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py")],
        cwd=str(ROOT),
        env=env,
        check=True,
    )


@pytest.fixture(scope="module")
def client(trained_model):
    import app as app_module
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def spark_session():
    """Shared SparkSession for unit tests."""
    sys.path.insert(0, str(ROOT))
    from src.spark_session import create_spark
    spark = create_spark("flood-test")
    yield spark
    spark.stop()


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_features_endpoint(client):
    r = client.get("/features")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body
    assert len(body["features"]) == 20


def test_predict_midrange(client):
    payload = {feat: 5 for feat in [
        "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
        "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
        "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
        "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
        "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
        "InadequatePlanning", "PoliticalFactors",
    ]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["flood_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert body["model_source"]


def test_predict_low_risk(client):
    """Low-value features should produce a lower probability than high-value ones."""
    low_payload  = {feat: 1 for feat in _all_features()}
    high_payload = {feat: 9 for feat in _all_features()}
    r_low  = client.post("/predict", json=low_payload).json()
    r_high = client.post("/predict", json=high_payload).json()
    assert r_low["flood_probability"] < r_high["flood_probability"]


def test_predict_boundary_values(client):
    """Min (0) and max (10) values should not crash the API."""
    for val in (0, 10):
        payload = {feat: val for feat in _all_features()}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200


def test_predict_missing_field(client):
    """Partial payload should return 422 validation error."""
    r = client.post("/predict", json={"MonsoonIntensity": 5})
    assert r.status_code == 422


def test_predict_out_of_range(client):
    """Values outside [0, 10] should return 422."""
    payload = {feat: 5 for feat in _all_features()}
    payload["MonsoonIntensity"] = 99
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# PySpark unit tests (no API, no pandas)
# ---------------------------------------------------------------------------

def test_load_training_data_pyspark(spark_session):
    """load_training_data returns a PySpark DataFrame with correct columns."""
    from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN
    from src.preprocessing import load_training_data

    df = load_training_data(spark_session, DATA_PATH, max_rows=500)
    assert df.count() > 0
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing column: {col}"


def test_pipeline_fit_transform_pyspark(spark_session):
    """Pipeline fits and transforms without errors; prediction column exists."""
    from src.config import DATA_PATH, TARGET_COLUMN
    from src.preprocessing import build_pipeline, load_training_data

    df = load_training_data(spark_session, DATA_PATH, max_rows=500)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    pipeline = build_pipeline()
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)
    assert "prediction" in preds.columns
    assert preds.count() > 0


def test_engineered_features_present(spark_session):
    """SQLTransformer adds all engineered columns."""
    from src.config import DATA_PATH
    from src.preprocessing import ENGINEERED_COLUMNS, build_pipeline, load_training_data

    df = load_training_data(spark_session, DATA_PATH, max_rows=200)
    pipeline = build_pipeline(use_engineered=True)
    model = pipeline.fit(df)
    result = model.transform(df)
    for col in ENGINEERED_COLUMNS:
        assert col in result.columns, f"Engineered column missing: {col}"


def test_predict_one_pyspark(spark_session):
    """predict_one returns a float in [0, 1] using PySpark only."""
    from src.prediction import load_model, predict_one

    model = load_model(spark_session)
    features = {feat: 5.0 for feat in _all_features()}
    prob = predict_one(model, spark_session, features)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_null_handling_pyspark(spark_session):
    """Rows with nulls are dropped by load_training_data."""
    from pyspark.sql import Row
    from src.config import FEATURE_COLUMNS, TARGET_COLUMN
    from src.preprocessing import load_training_data

    df = load_training_data(spark_session, str(
        ROOT / "flood data" / "train.csv"
    ), max_rows=1000)
    # After loading, no nulls should remain
    from pyspark.sql import functions as F
    null_count = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in FEATURE_COLUMNS + [TARGET_COLUMN]
    ]).collect()[0].asDict()
    assert all(v == 0 for v in null_count.values()), f"Nulls found: {null_count}"


def test_correlation_with_target_pyspark(spark_session):
    """Pearson correlation between features and target is computable via PySpark."""
    from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN
    from src.preprocessing import load_training_data

    df = load_training_data(spark_session, DATA_PATH, max_rows=500)
    for feat in FEATURE_COLUMNS[:5]:  # spot-check first 5
        corr = df.stat.corr(feat, TARGET_COLUMN, method="pearson")
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_features() -> list[str]:
    return [
        "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
        "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
        "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
        "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
        "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
        "InadequatePlanning", "PoliticalFactors",
    ]
