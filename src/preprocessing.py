"""
PySpark ML preprocessing pipeline.

All data transformations — loading, cleaning, feature engineering,
scaling, and model assembly — are done entirely with PySpark.
No pandas is used anywhere in this module.
"""
import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, StandardScaler, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

# ---------------------------------------------------------------------------
# Derived / engineered feature names (computed via PySpark SQL)
# ---------------------------------------------------------------------------
ENGINEERED_COLUMNS = [
    "risk_sum",          # sum of all raw risk factors
    "risk_mean",         # mean of all raw risk factors
    "infra_risk",        # infrastructure sub-score
    "env_risk",          # environmental sub-score
    "mgmt_risk",         # management / governance sub-score
]

ALL_MODEL_FEATURES = FEATURE_COLUMNS + ENGINEERED_COLUMNS


def _feature_engineering_sql() -> str:
    """
    Return a SQLTransformer SQL string that adds engineered columns.
    All arithmetic is done inside Spark SQL — no Python loops at runtime.
    """
    raw_sum  = " + ".join(f"`{c}`" for c in FEATURE_COLUMNS)
    raw_mean = f"({raw_sum}) / {len(FEATURE_COLUMNS)}"

    # Infrastructure sub-group
    infra_cols = [
        "DamsQuality", "DrainageSystems", "DeterioratingInfrastructure",
        "IneffectiveDisasterPreparedness",
    ]
    infra_expr = " + ".join(f"`{c}`" for c in infra_cols)

    # Environmental sub-group
    env_cols = [
        "Deforestation", "Siltation", "WetlandLoss",
        "CoastalVulnerability", "Landslides", "Watersheds",
    ]
    env_expr = " + ".join(f"`{c}`" for c in env_cols)

    # Management / governance sub-group
    mgmt_cols = [
        "RiverManagement", "Urbanization", "AgriculturalPractices",
        "Encroachments", "InadequatePlanning", "PoliticalFactors",
    ]
    mgmt_expr = " + ".join(f"`{c}`" for c in mgmt_cols)

    return (
        f"SELECT *, "
        f"CAST(({raw_sum}) AS DOUBLE) AS risk_sum, "
        f"CAST(({raw_mean}) AS DOUBLE) AS risk_mean, "
        f"CAST(({infra_expr}) AS DOUBLE) AS infra_risk, "
        f"CAST(({env_expr})   AS DOUBLE) AS env_risk, "
        f"CAST(({mgmt_expr})  AS DOUBLE) AS mgmt_risk "
        f"FROM __THIS__"
    )


def build_pipeline(use_engineered: bool = True) -> Pipeline:
    """
    Build a PySpark ML Pipeline:
      1. SQLTransformer  — adds engineered features (pure Spark SQL)
      2. VectorAssembler — assembles feature vector
      3. StandardScaler  — zero-mean, unit-variance scaling
      4. RandomForestRegressor
    """
    feature_cols = ALL_MODEL_FEATURES if use_engineered else FEATURE_COLUMNS

    stages = []

    if use_engineered:
        stages.append(
            SQLTransformer(statement=_feature_engineering_sql())
        )

    stages += [
        VectorAssembler(inputCols=feature_cols, outputCol="features_raw"),
        StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True,
        ),
        RandomForestRegressor(
            featuresCol="features",
            labelCol=TARGET_COLUMN,
            numTrees=int(os.getenv("RF_NUM_TREES", "80")),
            maxDepth=int(os.getenv("RF_MAX_DEPTH", "12")),
            minInstancesPerNode=int(os.getenv("RF_MIN_INSTANCES", "2")),
            featureSubsetStrategy=os.getenv("RF_FEATURE_SUBSET", "auto"),
            seed=42,
        ),
    ]

    return Pipeline(stages=stages)


def load_training_data(spark: SparkSession, data_path, max_rows: int | None) -> DataFrame:
    """
    Load CSV with PySpark, cast all feature/target columns to DoubleType,
    drop nulls, and optionally limit rows.  No pandas involved.
    """
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(data_path))
    )

    # Cast every required column to Double explicitly (handles string-typed CSVs)
    cast_exprs = [
        F.col(c).cast(DoubleType()).alias(c)
        for c in FEATURE_COLUMNS + [TARGET_COLUMN]
        if c in df.columns
    ]
    df = df.select(cast_exprs).dropna()

    if max_rows and max_rows > 0:
        df = df.limit(max_rows)

    return df


def compute_class_weights(df: DataFrame) -> dict[str, float]:
    """
    Compute inverse-frequency weights for Low / Medium / High risk buckets
    using PySpark aggregations — useful for weighted training or evaluation.
    """
    total = df.count()
    bucket_df = df.select(
        F.when(F.col(TARGET_COLUMN) < 0.45, "Low")
         .when(F.col(TARGET_COLUMN) < 0.55, "Medium")
         .otherwise("High")
         .alias("bucket")
    )
    counts = {
        row["bucket"]: row["count"]
        for row in bucket_df.groupBy("bucket").count().collect()
    }
    return {bucket: total / cnt for bucket, cnt in counts.items()}
