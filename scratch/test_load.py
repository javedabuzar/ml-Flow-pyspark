import os
import yaml
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.spark

# Set env vars like in run_app.bat
os.environ["JAVA_HOME"] = r"d:\ml flow abuzar"
os.environ["HADOOP_HOME"] = r"C:\hardoop"
os.environ["PYSPARK_PYTHON"] = r"d:\ml_flow_venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"d:\ml_flow_venv\Scripts\python.exe"
os.environ["PATH"] = rf"{os.environ['JAVA_HOME']}\bin;{os.environ['HADOOP_HOME']}\bin;" + os.environ["PATH"]

try:
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("TestLoad") \
        .master("local[1]") \
        .config("spark.driver.memory", "512m") \
        .getOrCreate()
    print("Spark initialized.")

    PREPROCESSOR_PATH = "src/models/preprocessing_pipeline_model"
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}...")
    preprocessor = PipelineModel.load(PREPROCESSOR_PATH)
    print("Preprocessor loaded.")

    print("Loading MLflow model...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "Flood_Prediction_Spark_Model"
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.spark.load_model(model_uri)
    print("MLflow model loaded.")

    print("Success!")
except Exception as e:
    print(f"FAILED: {e}")
finally:
    try:
        spark.stop()
    except:
        pass
