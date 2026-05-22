import os
import yaml
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.spark

# Load params
params = yaml.safe_load(open("params.yaml"))

def predict():
    spark = SparkSession.builder \
        .appName(f"{params['spark']['app_name']}_Prediction") \
        .master(params['spark']['master']) \
        .getOrCreate()

    # 1. Model Load
    model_name = "Flood_Prediction_Spark_Model"
    model_uri = f"models:/{model_name}/latest"
    
    try:
        model = mlflow.spark.load_model(model_uri)
        print(f"Model {model_name} Loaded!")
    except Exception as e:
        # Fallback to local run if registry is empty
        print(f"Registry load failed, trying local run artifacts... ({e})")
        # You'd typically find the latest run ID here
        return

    # 2. Preprocessing Pipeline Model Load
    pipeline_model_path = "src/models/preprocessing_pipeline_model"
    preprocessing_model = PipelineModel.load(pipeline_model_path)
    print("Preprocessing Pipeline Model Loaded!")

    # 3. Sample Data for Prediction
    # Get schema from raw data to match columns exactly
    raw_df = spark.read.parquet(params["data"]["raw_path"])
    input_cols = [c for c in raw_df.columns if c not in ["id", "FloodProbability"]]
    
    # Create sample data with default values (e.g., 5.0)
    sample_values = [5.0] * len(input_cols)
    new_data = spark.createDataFrame([tuple(sample_values)], input_cols)
    
    print("\nNew Data (Sample):")
    new_data.show()

    # 4. Preprocess data using the saved pipeline model
    new_data_transformed = preprocessing_model.transform(new_data)

    # 5. Prediction
    predictions = model.transform(new_data_transformed)

    print("\n--- Predictions ---")
    predictions.select("prediction").show()

    spark.stop()

if __name__ == "__main__":
    predict()