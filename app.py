from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructField, StructType
from pyspark.ml import PipelineModel
import mlflow
import mlflow.spark
import glob
import os
import sys
import yaml
from dotenv import load_dotenv
from threading import Lock

load_dotenv()

# Ensure PySpark workers use the same Python as this process
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

# Load params
params = yaml.safe_load(open("params.yaml"))

app = FastAPI(title="Flood Prediction API", description="PySpark + MLflow powered API for Flood Probability")

# Globals for lazy loading
APP_NAME = "FloodPredictionAPI"
spark = None
preprocessor = None
model = None
model_type = None  # 'spark' or 'pyfunc'
_models_lock = Lock()
_last_load_spark = None  # track cache mode so /health cannot poison /predict
_feature_dtypes_cache = None
MODEL_NAME = "Flood_Prediction_Spark_Model"
TARGET_COL = "FloodProbability"
PREPROCESSOR_PATH = "src/models/preprocessing_pipeline_model"
_default_spark_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "spark")
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", _default_spark_dir)


def _use_spark_enabled() -> bool:
    return os.getenv("USE_SPARK", "1").lower() not in ("0", "false", "no")


def _spark_is_alive() -> bool:
    global spark
    if spark is None:
        return False
    try:
        return not spark.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False


def get_spark():
    global spark
    if spark is not None and not _spark_is_alive():
        try:
            spark.stop()
        except Exception:
            pass
        spark = None

    if spark is None:
        on_railway = bool(os.getenv("PORT"))
        driver_mem = os.getenv(
            "SPARK_DRIVER_MEMORY", "512m" if on_railway else "1g"
        )
        executor_mem = os.getenv(
            "SPARK_EXECUTOR_MEMORY", "256m" if on_railway else "1g"
        )
        os.makedirs(SPARK_LOCAL_DIR, exist_ok=True)
        # Single worker in containers avoids driver/worker connection refused on Railway
        master = os.getenv("SPARK_MASTER", params.get("spark", {}).get("master", "local[1]"))
        if on_railway and master in ("local[*]", "local"):
            master = "local[1]"
        spark = (
            SparkSession.builder.appName(APP_NAME)
            .master(master)
            .config("spark.driver.memory", driver_mem)
            .config("spark.executor.memory", executor_mem)
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.local.dir", SPARK_LOCAL_DIR)
            .config("spark.network.timeout", "800s")
            .config("spark.executor.heartbeatInterval", "100s")
            .config("spark.python.worker.reuse", "true")
            .config("spark.python.worker.timeout", "120")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "1")
            .getOrCreate()
        )
    return spark


def reset_spark_and_models():
    global spark, preprocessor, model, model_type, _last_load_spark
    with _models_lock:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass
        spark = None
        preprocessor = None
        model = None
        model_type = None
        _last_load_spark = None


def _clear_model_cache_if_mode_changed(load_spark: bool):
    global preprocessor, model, model_type, _last_load_spark
    if _last_load_spark is not None and _last_load_spark != load_spark:
        preprocessor = None
        model = None
        model_type = None
    _last_load_spark = load_spark


def _is_spark_session_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "no active or default spark session" in msg
        or "spark session" in msg
        or "connection refused" in msg
        or "errno 111" in msg
        or "jvm is not running" in msg
    )


def _spark_artifacts_present() -> bool:
    has_spark_model = bool(glob.glob("mlruns/**/artifacts/model", recursive=True))
    return has_spark_model or os.path.exists(PREPROCESSOR_PATH)

def _pyfunc_loader_module(artifacts_dir: str) -> str | None:
    mlmodel_path = os.path.join(artifacts_dir, "MLmodel")
    if not os.path.exists(mlmodel_path):
        return None
    try:
        with open(mlmodel_path) as f:
            doc = yaml.safe_load(f) or {}
        pf = (doc.get("flavors") or {}).get("python_function") or {}
        return pf.get("loader_module")
    except Exception:
        return None


def _pyfunc_artifact_sort_key(artifacts_dir: str, load_spark: bool = True) -> tuple:
    """PySpark-first when load_spark=True; sklearn-first when Spark is disabled."""
    loader = _pyfunc_loader_module(artifacts_dir) or ""
    if load_spark:
        if loader == "mlflow.spark":
            rank = 0
        elif loader == "mlflow.sklearn":
            rank = 2
        else:
            rank = 1
    else:
        if loader == "mlflow.sklearn":
            rank = 0
        elif loader == "mlflow.spark":
            rank = 2
        else:
            rank = 1
    return (rank, -os.path.getmtime(artifacts_dir))


def load_models(load_spark=None):
    global preprocessor, model
    if load_spark is None:
        load_spark = _use_spark_enabled()
    with _models_lock:
        _clear_model_cache_if_mode_changed(load_spark)
        if load_spark:
            get_spark()
        if preprocessor is None:
            if load_spark and os.path.exists(PREPROCESSOR_PATH):
                try:
                    get_spark()
                    preprocessor = PipelineModel.load(PREPROCESSOR_PATH)
                    print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
                except Exception as prep_err:
                    print(f"Preprocessor load failed: {prep_err}")
            elif os.path.exists(PREPROCESSOR_PATH):
                print(f"Preprocessor file exists but Spark loading is disabled for this call: {PREPROCESSOR_PATH}")
            else:
                print(f"Preprocessor path not found: {PREPROCESSOR_PATH}")
        if model is None:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            model_uri = f"models:/{MODEL_NAME}/latest"
            if load_spark:
                # Try Spark model from registry first (preferred)
                try:
                    get_spark()
                    model = mlflow.spark.load_model(model_uri)
                    globals()['model_type'] = 'spark'
                    print("Model loaded from registry (Spark)")
                except Exception as registry_error:
                    print(f"Registry Spark model load failed: {registry_error}")
                    # Try local Spark model directories under mlruns/**/artifacts/model
                    spark_model_paths = sorted(
                        glob.glob("mlruns/**/artifacts/model", recursive=True),
                        key=os.path.getmtime,
                        reverse=True
                    )
                    if spark_model_paths:
                        fallback_model_path = os.path.abspath(spark_model_paths[0])
                        try:
                            get_spark()
                            model = mlflow.spark.load_model(fallback_model_path)
                            globals()['model_type'] = 'spark'
                            print(f"Model loaded from local Spark artifact: {fallback_model_path}")
                        except Exception as spark_local_err:
                            print(f"Local Spark load failed: {spark_local_err}")
                    else:
                        print("No local Spark model artifact found under mlruns/**/artifacts/model")

            # If registry/local Spark both failed or Spark loading skipped, try pyfunc artifacts in mlruns
            if model is None:
                artifact_dirs = [
                    p for p in glob.glob("mlruns/**/artifacts", recursive=True)
                    if os.path.exists(os.path.join(p, "MLmodel"))
                ]
                artifact_dirs.sort(key=lambda p: _pyfunc_artifact_sort_key(p, load_spark))
                for p in artifact_dirs:
                    loader = _pyfunc_loader_module(p)
                    if loader == "mlflow.sklearn" and load_spark:
                        continue
                    if loader == "mlflow.spark" and not load_spark:
                        continue
                    if loader == "mlflow.spark":
                        try:
                            get_spark()
                            model = mlflow.spark.load_model(p)
                            globals()["model_type"] = "spark"
                            print(f"Spark model loaded from artifact path: {p}")
                            break
                        except Exception as spark_err:
                            print(f"Spark artifact load failed for {p}: {spark_err}")
                            continue
                    try:
                        model = mlflow.pyfunc.load_model(p)
                        globals()["model_type"] = "pyfunc"
                        print(f"Fallback: model loaded as pyfunc from {p}")
                        break
                    except Exception as err:
                        print(f"pyfunc load failed for {p}: {err}")
                        continue

    return preprocessor, model

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionInput(BaseModel):
    MonsoonIntensity: float
    TopographyDrainage: float
    RiverManagement: float
    Deforestation: float
    Urbanization: float
    ClimateChange: float
    DamsQuality: float
    Siltation: float
    AgriculturalPractices: float
    Encroachments: float
    IneffectiveDisasterPreparedness: float
    DrainageSystems: float
    CoastalVulnerability: float
    Landslides: float
    Watersheds: float
    DeterioratingInfrastructure: float
    PopulationScore: float
    WetlandLoss: float
    InadequatePlanning: float
    PoliticalFactors: float


def _feature_names() -> list[str]:
    return list(PredictionInput.model_fields.keys())


def _get_raw_feature_dtypes() -> dict:
    global _feature_dtypes_cache
    if _feature_dtypes_cache is not None:
        return _feature_dtypes_cache

    # Training notebook uses string feature columns; API sliders must match ("5" not 5.0)
    dtypes = {name: StringType() for name in _feature_names()}

    _feature_dtypes_cache = dtypes
    return dtypes


def _coerce_feature_value(val: float, dtype):
    # Training data uses string categoricals; always stringify slider ints first
    if isinstance(dtype, (StringType, IntegerType)):
        return str(int(round(val)))
    if isinstance(dtype, DoubleType):
        return float(val)
    return str(int(round(val)))


def _build_spark_input_df(sp, input_dict: dict):
    dtypes = _get_raw_feature_dtypes()
    row = [_coerce_feature_value(input_dict[name], dtypes[name]) for name in _feature_names()]
    schema = StructType(
        [StructField(name, dtypes[name], True) for name in _feature_names()]
    )
    return sp.createDataFrame([row], schema=schema)


def _normalize_probability(value: float) -> float:
    v = float(value)
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


@app.get("/")
def home():
    return FileResponse("static/index.html")

def _run_prediction(pr, mdl, input_dict: dict) -> float:
    mtype = globals().get("model_type")
    if mtype == "spark":
        sp = get_spark()
        input_df = _build_spark_input_df(sp, input_dict)
        print(f"Input DF columns: {input_df.columns}, dtypes: {input_df.dtypes}")

        if pr is None:
            raise RuntimeError("Preprocessor not found for Spark model")

        transformed_df = pr.transform(input_df)
        print("Preprocessing done.")

        prediction_df = mdl.transform(transformed_df)
        print("Prediction transformation done.")

        row = prediction_df.select("prediction").first()
        if row is None:
            raise ValueError("Model returned no prediction")
        return _normalize_probability(float(row[0]))

    import pandas as pd

    dtypes = _get_raw_feature_dtypes()
    row = {
        name: _coerce_feature_value(input_dict[name], dtypes[name])
        for name in _feature_names()
    }
    py_input = pd.DataFrame([row])
    preds = mdl.predict(py_input)
    if hasattr(preds, "__len__"):
        return _normalize_probability(float(preds[0]))
    return _normalize_probability(float(preds))


@app.post("/predict")
def predict(data: PredictionInput):
    load_spark = _use_spark_enabled()
    input_dict = {k: float(v) for k, v in data.model_dump().items()}
    last_error = None

    for attempt in range(3):
        try:
            if attempt > 0:
                reset_spark_and_models()

            pr, mdl = load_models(load_spark=load_spark)
            if mdl is None:
                print("Model not available — returning fallback prediction")
                return {
                    "flood_probability": 0.5,
                    "status": "fallback",
                }

            if globals().get("model_type") == "spark":
                get_spark()

            result = _run_prediction(pr, mdl, input_dict)
            print(f"Prediction successful: {result}")
            return {
                "flood_probability": round(result, 4),
                "status": "success",
            }
        except Exception as e:
            last_error = e
            print(f"Prediction Error (attempt {attempt + 1}): {e}")
            import traceback

            traceback.print_exc()
            if attempt < 2 and load_spark and _is_spark_session_error(e):
                print(f"Retrying prediction after Spark reset (attempt {attempt + 1})...")
                continue
            raise HTTPException(status_code=500, detail=str(last_error))


@app.get("/health")
def health():
    spark_ready = (
        _spark_artifacts_present()
        and os.path.exists(PREPROCESSOR_PATH)
    )
    return {
        "app": True,
        "use_spark": _use_spark_enabled(),
        "spark_artifacts_present": _spark_artifacts_present(),
        "preprocessor_present": os.path.exists(PREPROCESSOR_PATH),
        "model_type": "spark" if spark_ready else None,
        "model_loaded": spark_ready,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)