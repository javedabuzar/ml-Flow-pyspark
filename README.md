# Flood Risk ML — PySpark + MLflow + Docker

End-to-end flood probability regression using **PySpark for everything** —
data loading, feature engineering, model training, evaluation, inference,
and EDA all run on PySpark. Experiment tracking with **MLflow**, served via
**FastAPI**, containerised with **Docker**, and automated with **GitHub Actions** CI/CD.

## Architecture

```
flood data/train.csv
        │
        ▼
src/preprocessing.py   ← PySpark: load CSV, cast types, drop nulls
        │                          SQLTransformer (feature engineering)
        │                          VectorAssembler + StandardScaler
        ▼
src/train.py           ← PySpark: fit RandomForestRegressor pipeline
        │                          RegressionEvaluator (RMSE / MAE / R²)
        │                          MLflow logging (params, metrics, model)
        ▼
models/spark_pipeline  ← saved PipelineModel (PySpark native format)
        │
        ▼
src/prediction.py      ← PySpark: single-row & batch inference
        │
        ▼
app.py                 ← FastAPI: /predict endpoint (calls PySpark)
        │
        ▼
static/index.html      ← Web UI
```

## Quick start (local)

```bash
pip install -r requirements.txt
# Java 17+ required for PySpark
set TRAIN_MAX_ROWS=30000
python src/train.py
python scripts/start.py
```

Open http://localhost:8000

## EDA (PySpark)

```bash
python src/eda.py
```

Runs full exploratory analysis — summary stats, quantiles, Pearson
correlations, outlier counts, risk bucket distribution — all via PySpark.

## Batch prediction (PySpark)

```bash
python src/prediction.py "flood data/test.csv"
```

Scores an entire CSV file using PySpark and prints summary statistics.

## Docker

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API + UI | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_MAX_ROWS` | 50000 | Limit rows for faster training |
| `RF_NUM_TREES` | 80 | Random Forest tree count |
| `RF_MAX_DEPTH` | 12 | Random Forest max depth |
| `RF_MIN_INSTANCES` | 2 | Min instances per node |
| `RF_FEATURE_SUBSET` | auto | Feature subset strategy |
| `TEST_FRACTION` | 0.2 | Train/test split ratio |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking path |
| `SPARK_DRIVER_MEMORY` | 2g | Spark driver memory |
| `SPARK_SHUFFLE_PARTITIONS` | 8 | Spark shuffle partitions |
| `SKIP_TRAIN` | — | Set `1` to skip auto-train on startup |
| `EDA_MAX_ROWS` | 50000 | Rows to use for EDA |

## Tests

```bash
pytest tests/ -v
```

Tests cover:
- API endpoints (`/health`, `/features`, `/predict`)
- Input validation (missing fields, out-of-range values)
- PySpark data loading and null handling
- PySpark pipeline fit/transform
- Engineered feature generation
- PySpark-based correlation computation
- End-to-end single-row inference

## Deploy to Render.com (free)

1. **Train model locally first** (one-time):
   ```bash
   set TRAIN_MAX_ROWS=30000
   python src/train.py
   ```
   This creates `models/spark_pipeline/` which gets baked into the Docker image.

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "add trained model + render config"
   git push
   ```

3. **On [Render.com](https://render.com)**:
   - New → Web Service → Connect GitHub repo
   - Runtime: **Docker**
   - Set these env vars:
     - `SKIP_TRAIN` = `1`
     - `SPARK_DRIVER_MEMORY` = `400m`
     - `SPARK_SHUFFLE_PARTITIONS` = `2`
     - `SPARK_UI_ENABLED` = `false`
   - Click **Deploy** → wait ~5 min for Docker build

4. App live at `https://your-app.onrender.com`

> **Note:** Render free tier sleeps after 15 min inactivity. First request after sleep takes ~30s (Spark JVM startup).

## CI/CD

`.github/workflows/mlflow_pyspark.yml` trains on a sample, runs pytest,
builds and smoke-tests the Docker image on every push/PR.
