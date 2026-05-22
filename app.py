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
            try:
                model = mlflow.spark.load_model(model_uri)
                print("Model loaded from registry")
            except Exception as registry_error:
                print(f"Registry model load failed: {registry_error}")
                local_model_paths = sorted(
                    glob.glob("mlruns/*/*/artifacts/model"),
                    key=os.path.getmtime,
                    reverse=True
                )
                if not local_model_paths:
                    print("No local Spark model artifact found in mlruns/*/*/artifacts/model")
                else:
                    fallback_model_path = local_model_paths[0]
                    print(f"Falling back to local Spark model: {fallback_model_path}")
                    model = mlflow.spark.load_model(fallback_model_path)
                    print("Model loaded from local artifact")

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
        # Ensure Spark and models are available (lazy load)
        sp = get_spark()
        pr, mdl = load_models()
        if pr is None or mdl is None:
            raise RuntimeError("Model or preprocessor not available. Check server logs.")
        # Use JVM-native JSON loading to avoid Python worker overhead
        import json
        import tempfile
        # Cast to int to match training data (integer features)
        input_dict = {k: int(v) for k, v in data.model_dump().items()}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([input_dict], f)
            temp_path = f.name.replace("\\", "/")
        
        input_df = spark.read.json(temp_path)
        print(f"Input DF loaded via JSON. Columns: {input_df.columns}")
        
        print(f"Input DF created. Columns: {input_df.columns}")
        
        # Preprocess
        transformed_df = preprocessor.transform(input_df)
        print("Preprocessing done.")
        
        # Predict
        prediction_df = model.transform(transformed_df)
        print("Prediction transformation done.")
        
        # Extract result
        # Using first() instead of collect() for single row
        row = prediction_df.select("prediction").first()
        if row is None:
            raise ValueError("Model returned no prediction")
        result = row[0]
        
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
    try:
        sp = get_spark()
        status["spark"] = True
    except Exception as e:
        status["spark"] = False
        status["spark_error"] = str(e)

    try:
        pr, mdl = load_models()
        status["model_loaded"] = (pr is not None and mdl is not None)
    except Exception as e:
        status["model_loaded"] = False
        status["model_error"] = str(e)

    return status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)