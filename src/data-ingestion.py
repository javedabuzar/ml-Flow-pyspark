import kagglehub
import os
from pyspark.sql import SparkSession
import yaml

# Load params
params = yaml.safe_load(open("params.yaml"))

def data_ingestion():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName(params["spark"]["app_name"]) \
        .master(params["spark"]["master"]) \
        .getOrCreate()

    raw_parquet_path = params["data"]["raw_path"]
    local_csv_path = "src/data/raw/flood.csv"

    if os.path.exists(raw_parquet_path) and os.listdir(raw_parquet_path):
        print("Raw parquet data already exists and is not empty. Skipping ingestion.")
        return
    
    csv_path = None
    if os.path.exists(local_csv_path):
        print(f"Found local CSV at {local_csv_path}. Using it.")
        csv_path = local_csv_path
    else:
        print("Data ingestion start (Downloading from Kaggle)...")
        try:
            # Download dataset using kagglehub
            path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
            csv_path = os.path.join(path, "flood.csv")
            print(f"Data Downloaded to: {csv_path}")
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            print(f"Please download flood.csv manually and place it at {local_csv_path}")
            spark.stop()
            return

    # Load into Spark DataFrame
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    print(f"Data Loaded — Count: {df.count()}")
    df.show(5)

    # Save as Parquet
    os.makedirs(os.path.dirname(raw_parquet_path), exist_ok=True)
    df.write.mode("overwrite").parquet(raw_parquet_path)
    
    print(f"Raw data saved to {raw_parquet_path}")
    spark.stop()

if __name__ == "__main__":
    data_ingestion()