from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.spark
import os
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()

# Ensure PySpark workers use the same Python as this process
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Load params
params = yaml.safe_load(open("params.yaml"))

app = FastAPI(title="Flood Prediction API", description="PySpark + MLflow powered API for Flood Probability")

# Initialize Spark Session
APP_NAME = "FloodPredictionAPI"
spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master("local") \
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

# Load Models
MODEL_NAME = "Flood_Prediction_Spark_Model"
PREPROCESSOR_PATH = "src/models/preprocessing_pipeline_model"

try:
    # Load preprocessing pipeline
    preprocessor = PipelineModel.load(PREPROCESSOR_PATH)
    
    # Load registered model from MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.spark.load_model(model_uri)
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # We don't raise here so the app can start, but endpoints will fail

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)