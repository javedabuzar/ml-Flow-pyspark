import os
import yaml
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from dotenv import load_dotenv

load_dotenv()

# Load params
params = yaml.safe_load(open("params.yaml"))

# Set MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Flood_Prediction_Spark_Notebook_Logic")

def train():
    spark = SparkSession.builder \
        .appName(f"{params['spark']['app_name']}_Training") \
        .master(params['spark']['master']) \
        .getOrCreate()

    # Enable MLflow autologging
    mlflow.pyspark.ml.autolog()

    # Load processed data
    train_df = spark.read.parquet(os.path.join(params["data"]["processed_path"], "train.parquet"))
    test_df = spark.read.parquet(os.path.join(params["data"]["processed_path"], "test.parquet"))

    print(f"Train Count: {train_df.count()} | Test Count: {test_df.count()}")

    # Model from Notebook: LinearRegression
    print("\nTraining Linear Regression (from notebook logic)...")
    
    with mlflow.start_run(run_name="Spark_LR_Notebook_Logic"):
        lr = LinearRegression(featuresCol="features", labelCol="label")
        model = lr.fit(train_df)
        
        # Evaluation
        predictions = model.transform(test_df)
        
        evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
        evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        
        r2 = evaluator_r2.evaluate(predictions)
        rmse = evaluator_rmse.evaluate(predictions)
        
        print(f"Linear Regression - R2: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Log to MLflow
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_rmse", rmse)
        
        # Save model
        mlflow.spark.log_model(model, artifact_path="model")

    print("\nTraining completed!")
    spark.stop()

if __name__ == "__main__":
    train()