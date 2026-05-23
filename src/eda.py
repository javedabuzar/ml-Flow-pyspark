"""
Exploratory Data Analysis using pure PySpark — no pandas.

Usage:
    python src/eda.py
"""
import os
import sys
from pathlib import Path

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from src.spark_session import create_spark  # noqa: E402


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def summary_stats(df: DataFrame) -> None:
    """Print count, mean, stddev, min, max for all numeric columns."""
    print("\n=== Summary Statistics ===")
    df.select(FEATURE_COLUMNS + [TARGET_COLUMN]).describe().show(truncate=False)


def null_counts(df: DataFrame) -> None:
    """Print null/missing value counts per column."""
    print("\n=== Null Value Counts ===")
    null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    df.select(null_exprs).show(truncate=False)


def row_count(df: DataFrame) -> None:
    print(f"\n=== Total Rows: {df.count()} ===")


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def target_distribution(df: DataFrame, buckets: int = 10) -> None:
    """Show histogram-style bucket counts for the target column."""
    print(f"\n=== Target Distribution ({TARGET_COLUMN}) ===")
    min_val, max_val = df.select(
        F.min(TARGET_COLUMN).cast(DoubleType()),
        F.max(TARGET_COLUMN).cast(DoubleType()),
    ).first()

    bucket_size = (max_val - min_val) / buckets
    df.select(
        F.floor((F.col(TARGET_COLUMN).cast(DoubleType()) - min_val) / bucket_size)
        .cast("int")
        .alias("bucket")
    ).groupBy("bucket").count().orderBy("bucket").show(buckets + 1, truncate=False)


def feature_quantiles(df: DataFrame) -> None:
    """Print 25th, 50th, 75th percentile for each feature using PySpark approxQuantile."""
    print("\n=== Feature Quantiles (p25 / p50 / p75) ===")
    quantiles = df.approxQuantile(FEATURE_COLUMNS, [0.25, 0.50, 0.75], relativeError=0.01)
    header = f"{'Feature':<40} {'p25':>8} {'p50':>8} {'p75':>8}"
    print(header)
    print("-" * len(header))
    for col_name, (p25, p50, p75) in zip(FEATURE_COLUMNS, quantiles):
        print(f"{col_name:<40} {p25:>8.3f} {p50:>8.3f} {p75:>8.3f}")


# ---------------------------------------------------------------------------
# Correlation analysis (PySpark MLlib)
# ---------------------------------------------------------------------------

def correlation_with_target(df: DataFrame) -> None:
    """Compute Pearson correlation of each feature with the target using PySpark."""
    print(f"\n=== Pearson Correlation with {TARGET_COLUMN} ===")
    rows = []
    for feat in FEATURE_COLUMNS:
        corr_val = df.stat.corr(feat, TARGET_COLUMN, method="pearson")
        rows.append((feat, round(corr_val, 4)))

    # Sort by absolute correlation descending
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"{'Feature':<40} {'Corr':>8}")
    print("-" * 50)
    for feat, corr_val in rows:
        bar = "█" * int(abs(corr_val) * 20)
        direction = "+" if corr_val >= 0 else "-"
        print(f"{feat:<40} {corr_val:>8.4f}  {direction}{bar}")


def feature_correlation_matrix(df: DataFrame) -> None:
    """Print the full feature-feature Pearson correlation matrix via PySpark MLlib."""
    print("\n=== Feature Correlation Matrix (Pearson) ===")
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="_corr_vec")
    vec_df = assembler.transform(df).select("_corr_vec")
    matrix = Correlation.corr(vec_df, "_corr_vec", method="pearson").collect()[0][0]
    values = matrix.toArray()

    # Header row
    short = [c[:8] for c in FEATURE_COLUMNS]
    print(f"{'':>12}" + "".join(f"{s:>10}" for s in short))
    for i, row_name in enumerate(FEATURE_COLUMNS):
        row_vals = "".join(f"{values[i][j]:>10.3f}" for j in range(len(FEATURE_COLUMNS)))
        print(f"{row_name[:12]:<12}{row_vals}")


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def outlier_counts(df: DataFrame, multiplier: float = 1.5) -> None:
    """Count IQR-based outliers per feature using PySpark approxQuantile."""
    print(f"\n=== Outlier Counts (IQR × {multiplier}) ===")
    quantiles = df.approxQuantile(FEATURE_COLUMNS, [0.25, 0.75], relativeError=0.01)
    print(f"{'Feature':<40} {'Outliers':>10} {'% of total':>12}")
    print("-" * 65)
    total = df.count()
    for col_name, (q1, q3) in zip(FEATURE_COLUMNS, quantiles):
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        count = df.filter((F.col(col_name) < lower) | (F.col(col_name) > upper)).count()
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"{col_name:<40} {count:>10} {pct:>11.2f}%")


# ---------------------------------------------------------------------------
# Class balance (risk buckets)
# ---------------------------------------------------------------------------

def risk_bucket_counts(df: DataFrame) -> None:
    """Show how many rows fall into Low / Medium / High flood risk buckets."""
    print("\n=== Risk Bucket Distribution ===")
    df.select(
        F.when(F.col(TARGET_COLUMN) < 0.45, "Low")
         .when(F.col(TARGET_COLUMN) < 0.55, "Medium")
         .otherwise("High")
         .alias("risk_level")
    ).groupBy("risk_level").count().orderBy("risk_level").show(truncate=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_eda(spark: SparkSession, max_rows: int = 50000) -> None:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_PATH))
        .select(*(FEATURE_COLUMNS + [TARGET_COLUMN]))
        .dropna()
        .limit(max_rows)
        .cache()
    )

    row_count(df)
    null_counts(df)
    summary_stats(df)
    target_distribution(df)
    feature_quantiles(df)
    correlation_with_target(df)
    feature_correlation_matrix(df)
    outlier_counts(df)
    risk_bucket_counts(df)

    df.unpersist()


if __name__ == "__main__":
    max_rows = int(os.getenv("EDA_MAX_ROWS", "50000"))
    spark = create_spark("flood-eda")
    try:
        run_eda(spark, max_rows=max_rows)
    finally:
        spark.stop()
