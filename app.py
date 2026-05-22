from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
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

def get_spark():
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName(APP_NAME) \
            .master(params.get('spark', {}).get('master', 'local')) \
            .config("spark.driver.memory", "512m") \
            .config("spark.executor.memory", "512m") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "100s") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.python.worker.timeout", "120") \
            .config("spark.python.worker.faulthandler.enabled", "true") \
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
            .config("spark.ui.enabled", "false") \
            .config("spark.ui.showConsoleProgress", "false") \
            .getOrCreate()
    return spark

def load_models():
    global preprocessor, model
    with _models_lock:
        if preprocessor is None:
            if os.path.exists(PREPROCESSOR_PATH):
                preprocessor = PipelineModel.load(PREPROCESSOR_PATH)
                print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
            else:
                print(f"Preprocessor path not found: {PREPROCESSOR_PATH}")
        if model is None:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            model_uri = f"models:/{MODEL_NAME}/latest"
            # Try Spark model from registry first
            try:
                model = mlflow.spark.load_model(model_uri)
                globals()['model_type'] = 'spark'
                print("Model loaded from registry (Spark)")
            except Exception as registry_error:
                print(f"Registry Spark model load failed: {registry_error}")
                # Try local spark artifact path
                # search recursively for any artifacts/model directories under mlruns
                local_model_paths = sorted(
                    glob.glob("mlruns/**/artifacts/model", recursive=True),
                    key=os.path.getmtime,
                    reverse=True
                )
                if local_model_paths:
                    fallback_model_path = local_model_paths[0]
                    # Try Spark flavor first
                    try:
                        model = mlflow.spark.load_model(fallback_model_path)
                        globals()['model_type'] = 'spark'
                        print(f"Model loaded from local Spark artifact: {fallback_model_path}")
                    except Exception as spark_local_err:
                        print(f"Local Spark load failed: {spark_local_err}")
                        # As fallback, try python/pyfunc flavor (no JVM)
                        try:
                            import mlflow.pyfunc
                            model = mlflow.pyfunc.load_model(fallback_model_path)
                            globals()['model_type'] = 'pyfunc'
                            print(f"Model loaded as pyfunc from: {fallback_model_path}")
                        except Exception as pyfunc_err:
                            print(f"Local pyfunc load failed: {pyfunc_err}")
                else:
                    print("No local model artifact found in mlruns/*/*/artifacts/model")

            # If registry failed and we didn't set model yet, try loading any pyfunc artifact in mlruns
            if model is None:
                # look for any mlruns model artifact (recursive) and try pyfunc
                any_local = glob.glob("mlruns/**/artifacts/model", recursive=True)
                for p in sorted(any_local, key=os.path.getmtime, reverse=True):
                    try:
                        import mlflow.pyfunc
                        model = mlflow.pyfunc.load_model(p)
                        globals()['model_type'] = 'pyfunc'
                        print(f"Fallback: model loaded as pyfunc from {p}")
                        break
                    except Exception:
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

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Ensure models are available (lazy load)
        pr, mdl = load_models()
        if mdl is None:
            raise RuntimeError("Model not available. Check server logs.")

        # Prepare input
        input_dict = {k: float(v) for k, v in data.model_dump().items()}

        # If model is a Spark model, use Spark pipeline
        if globals().get('model_type') == 'spark':
            sp = get_spark()
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
            result = row[0]

        else:
            # Try pyfunc / sklearn-like model (no Spark)
            try:
                import pandas as pd
                py_input = pd.DataFrame([input_dict])
                # mlflow pyfunc models expose predict
                preds = mdl.predict(py_input)
                # handle array or scalar
                if hasattr(preds, '__len__'):
                    result = float(preds[0])
                else:
                    result = float(preds)
            except Exception as e:
                raise RuntimeError(f"Pyfunc prediction failed: {e}")

        print(f"Prediction successful: {result}")

        return {
            "flood_probability": round(float(result), 4),
            "status": "success"
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    status = {"app": True}
    # Do not start Spark here — starting Spark can crash on small containers.
    # Instead, only check model availability (pyfunc or spark) without forcing JVM.
    try:
        pr, mdl = load_models()
        status["model_loaded"] = (pr is not None and mdl is not None)
        status["model_type"] = globals().get('model_type')
    except Exception as e:
        status["model_loaded"] = False
        status["model_error"] = str(e)

    return status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)