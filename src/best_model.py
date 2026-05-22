import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()

# MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def register_best_model():
    client = MlflowClient()
    experiment_name = "Flood_Prediction_Spark_Notebook_Logic"
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} not found.")
        return

    # Search for the best run based on test_r2 metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_r2 DESC"]
    )

    if not runs:
        print("No runs found in experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_r2 = best_run.data.metrics.get("test_r2", 0)

    print(f"Best Run ID : {best_run_id}")
    print(f"Best Test R2: {best_r2:.4f}")

    # Register the model
    model_name = "Flood_Prediction_Spark_Model"
    mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name
    )

    print(f"Best Model Registered as {model_name}!")

if __name__ == "__main__":
    register_best_model()