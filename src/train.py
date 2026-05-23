"""Train flood probability model with PySpark and log to MLflow.

All data operations — loading, splitting, metrics, signature inference —
are done entirely with PySpark DataFrames. No pandas is used.
"""
import os
import sys
from pathlib import Path

import mlflow
from mlflow.types.schema import ColSpec, Schema
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_PATH,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    mlflow_tracking_uri,
)
from src.preprocessing import ALL_MODEL_FEATURES, build_pipeline, load_training_data  # noqa: E402
from src.spark_session import create_spark  # noqa: E402


def _regression_metrics(predictions: DataFrame) -> dict[str, float]:
    """Compute RMSE, MAE, R2 using PySpark evaluators — no pandas."""
    evaluator_rmse = RegressionEvaluator(
        labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="r2"
    )
    return {
        "rmse": float(evaluator_rmse.evaluate(predictions)),
        "mae": float(evaluator_mae.evaluate(predictions)),
        "r2": float(evaluator_r2.evaluate(predictions)),
    }


def _build_mlflow_signature() -> mlflow.models.ModelSignature:
    """Build MLflow signature from schema definitions — no pandas, no toPandas()."""
    input_schema = Schema([ColSpec("double", col) for col in FEATURE_COLUMNS])
    output_schema = Schema([ColSpec("double", "prediction")])
    return mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)


def train() -> str:
    max_rows = int(os.getenv("TRAIN_MAX_ROWS", "50000"))
    test_fraction = float(os.getenv("TEST_FRACTION", "0.2"))
    use_engineered = os.getenv("USE_ENGINEERED_FEATURES", "1") not in ("0", "false", "no")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    spark = create_spark("flood-train")
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

        df = load_training_data(spark, DATA_PATH, max_rows)
        train_df, test_df = df.randomSplit([1.0 - test_fraction, test_fraction], seed=42)
        pipeline = build_pipeline(use_engineered=use_engineered)

        with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "pyspark-rf-train")) as run:
            mlflow.log_param("train_max_rows", max_rows)
            mlflow.log_param("test_fraction", test_fraction)
            mlflow.log_param("num_features", len(FEATURE_COLUMNS))
            mlflow.log_param("use_engineered_features", use_engineered)
            mlflow.log_param("model_type", "RandomForestRegressor")

            model = pipeline.fit(train_df)
            train_predictions = model.transform(train_df)
            test_predictions = model.transform(test_df)

            train_metrics = _regression_metrics(train_predictions)
            test_metrics = _regression_metrics(test_predictions)
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)

            # Signature and input_example built from PySpark — no toPandas()
            signature = _build_mlflow_signature()
            sample_row = test_df.select(FEATURE_COLUMNS).limit(1).collect()[0]
            input_example = {col: float(sample_row[col]) for col in FEATURE_COLUMNS}

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                signature=signature,
                input_example=input_example,
            )

            # Save local copy for fast API startup (skip on Windows NativeIO errors)
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                model.write().overwrite().save(str(MODELS_DIR / "spark_pipeline"))
                print(f"Local model saved to {MODELS_DIR / 'spark_pipeline'}")
            except Exception as save_err:
                print(f"Warning: local model save skipped ({save_err}). "
                      "API will load from MLflow on startup.")

            run_id = run.info.run_id
            print(f"Training complete. MLflow run_id={run_id}")
            print(f"Test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}")
            return run_id
    finally:
        spark.stop()


if __name__ == "__main__":
    train()
