from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession
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
MODEL_NAME = "Flood_Prediction_Spark_Model"
PREPROCESSOR_PATH = "src/models/preprocessing_pipeline_model"
_default_spark_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "spark")
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", _default_spark_dir)


def _use_spark_enabled() -> bool:
    return os.getenv("USE_SPARK", "1").lower() not in ("0", "false", "no")


def get_spark():
    global spark
    if spark is None:
        driver_mem = os.getenv("SPARK_DRIVER_MEMORY", "1g")
        executor_mem = os.getenv("SPARK_EXECUTOR_MEMORY", "1g")
        os.makedirs(SPARK_LOCAL_DIR, exist_ok=True)
        master = params.get("spark", {}).get("master", "local[*]")
        spark = (
            SparkSession.builder.appName(APP_NAME)
            .master(master)
            .config("spark.driver.memory", driver_mem)
            .config("spark.executor.memory", executor_mem)
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.local.dir", SPARK_LOCAL_DIR)
            .config("spark.network.timeout", "800s")
            .config("spark.executor.heartbeatInterval", "100s")
            .config("spark.python.worker.reuse", "false")
            .config("spark.python.worker.timeout", "120")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
    return spark


def reset_spark_and_models():
    global spark, preprocessor, model, model_type
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


def _is_spark_session_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "no active or default spark session" in msg or "spark session" in msg


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
        if load_spark:
            get_spark()
        if preprocessor is None:
            if load_spark and os.path.exists(PREPROCESSOR_PATH):
                get_spark()
                preprocessor = PipelineModel.load(PREPROCESSOR_PATH)
                print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
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
                    if loader == "mlflow.spark" and not load_spark:
                        continue
                    if loader == "mlflow.spark":
                        try:
                            get_spark()
                        except Exception as spark_err:
                            print(f"Skipping Spark pyfunc artifact (no session): {p}: {spark_err}")
                            continue
                    try:
                        model = mlflow.pyfunc.load_model(p)
                        globals()['model_type'] = 'pyfunc'
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

@app.get("/")
def home():
    return FileResponse("static/index.html")

def _run_prediction(pr, mdl, input_dict: dict) -> float:
    mtype = globals().get("model_type")
    if mtype == "spark":
        sp = get_spark()
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([input_dict], f)
            temp_path = f.name.replace("\\", "/")

        input_df = sp.read.json(temp_path)
        print(f"Input DF loaded via JSON. Columns: {input_df.columns}")

        if pr is None:
            raise RuntimeError("Preprocessor not found for Spark model")

        transformed_df = pr.transform(input_df)
        print("Preprocessing done.")

        prediction_df = mdl.transform(transformed_df)
        print("Prediction transformation done.")

        row = prediction_df.select("prediction").first()
        if row is None:
            raise ValueError("Model returned no prediction")
        return float(row[0])

    import pandas as pd

    py_input = pd.DataFrame([input_dict])
    preds = mdl.predict(py_input)
    if hasattr(preds, "__len__"):
        return float(preds[0])
    return float(preds)


@app.post("/predict")
def predict(data: PredictionInput):
    load_spark = _use_spark_enabled()
    input_dict = {k: float(v) for k, v in data.model_dump().items()}
    last_error = None

    for attempt in range(2):
        try:
            if attempt > 0:
                reset_spark_and_models()

            pr, mdl = load_models(load_spark=load_spark)
            if mdl is None:
                print("Model not available — returning fallback prediction")
                return {"flood_probability": 0.5, "status": "fallback"}

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
            if attempt == 0 and load_spark and _is_spark_session_error(e):
                print("Retrying prediction after Spark session reset...")
                continue
            raise HTTPException(status_code=500, detail=str(last_error))


@app.get("/health")
def health():
    status = {
        "app": True,
        "use_spark": _use_spark_enabled(),
        "spark_artifacts_present": _spark_artifacts_present(),
        "preprocessor_present": os.path.exists(PREPROCESSOR_PATH),
    }
    # Do not start Spark here — starting Spark can crash on small containers.
    try:
        pr, mdl = load_models(load_spark=False)
        mtype = globals().get("model_type")
        status["model_type"] = mtype
        if mdl is None:
            status["model_loaded"] = False
        elif mtype == "spark":
            status["model_loaded"] = pr is not None
        else:
            status["model_loaded"] = True
    except Exception as e:
        status["model_loaded"] = False
        status["model_error"] = str(e)

    return status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)