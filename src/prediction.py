"""Model loading and inference — entirely via PySpark. No pandas used."""
import os

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

from src.config import FEATURE_COLUMNS, MODELS_DIR, REGISTERED_MODEL_NAME, mlflow_tracking_uri


# ---------------------------------------------------------------------------
# Model URI helpers
# ---------------------------------------------------------------------------

def _model_uri() -> str:
    uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
    if uri:
        return uri
    return f"models:/{REGISTERED_MODEL_NAME}/latest"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(spark: SparkSession) -> PipelineModel:
    """
    Load the trained PySpark PipelineModel.
    Priority:
      1. Local saved model at models/spark_pipeline  (fastest startup)
      2. MLflow model registry  (MLFLOW_MODEL_URI or registered name)
      3. Latest MLflow run artifact  (fallback when registry is empty)
    """
    local_path = MODELS_DIR / "spark_pipeline"
    if local_path.exists():
        return PipelineModel.load(str(local_path))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    uri = _model_uri()
    try:
        return mlflow.spark.load_model(uri, spark_session=spark)
    except Exception:
        # Fallback: latest run artifact when registry is empty
        runs = mlflow.search_runs(
            experiment_names=["flood-risk-pyspark"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError("No trained model found. Run: python src/train.py") from None
        run_id = runs.iloc[0]["run_id"]
        return mlflow.spark.load_model(f"runs:/{run_id}/model", spark_session=spark)


# ---------------------------------------------------------------------------
# Single-row inference  (used by the FastAPI /predict endpoint)
# ---------------------------------------------------------------------------

def predict_one(model: PipelineModel, spark: SparkSession, features: dict[str, float]) -> float:
    """
    Run inference on a single feature dict.

    Only the raw FEATURE_COLUMNS are needed as input — the PipelineModel
    itself contains the SQLTransformer stage that computes any engineered
    columns internally, so we never need to pass them in.

    Returns a scalar float probability.
    """
    schema = StructType([
        StructField(col, DoubleType(), nullable=False)
        for col in FEATURE_COLUMNS
    ])
    row = {col: float(features[col]) for col in FEATURE_COLUMNS}
    df = spark.createDataFrame([Row(**row)], schema=schema)
    prediction = (
        model.transform(df)
        .select("prediction")
        .collect()[0][0]
    )
    return float(prediction)


# ---------------------------------------------------------------------------
# Batch inference  (CLI: python src/prediction.py <csv_path>)
# ---------------------------------------------------------------------------

def predict_batch(model: PipelineModel, spark: SparkSession, csv_path: str) -> None:
    """
    Score an entire CSV file with PySpark and print summary statistics.
    All aggregations are done in Spark — no pandas.

    The CSV must contain the 20 raw feature columns (header row required).
    Engineered features are computed inside the pipeline automatically.
    """
    # Load only the raw feature columns — pipeline handles the rest
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
        .select([F.col(c).cast(DoubleType()).alias(c) for c in FEATURE_COLUMNS])
        .dropna()
    )

    predictions = model.transform(df).select(
        F.col("prediction").alias("flood_probability"),
        F.when(F.col("prediction") < 0.45, "Low")
         .when(F.col("prediction") < 0.55, "Medium")
         .otherwise("High")
         .alias("risk_level"),
        *FEATURE_COLUMNS,
    )

    total = predictions.count()
    print(f"\n{'='*60}")
    print(f"  BATCH PREDICTION RESULTS  ({csv_path})")
    print(f"  Total rows scored: {total:,}")
    print(f"{'='*60}")

    print("\n--- Flood Probability Statistics ---")
    predictions.select("flood_probability").describe().show(truncate=False)

    print("--- Risk Level Distribution ---")
    (
        predictions
        .groupBy("risk_level")
        .count()
        .withColumn("pct", F.round(F.col("count") / total * 100, 1))
        .orderBy("risk_level")
        .show(truncate=False)
    )

    print("--- Sample Predictions (top 10) ---")
    predictions.select(
        "flood_probability", "risk_level",
        *FEATURE_COLUMNS[:5],
    ).show(10, truncate=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.spark_session import create_spark

    if len(sys.argv) < 2:
        print("Usage: python src/prediction.py <path_to_csv>")
        print("Example: python src/prediction.py \"flood data/test.csv\"")
        sys.exit(1)

    csv_path = sys.argv[1]
    spark = create_spark("flood-batch-predict")
    try:
        loaded_model = load_model(spark)
        predict_batch(loaded_model, spark, csv_path)
    finally:
        spark.stop()
