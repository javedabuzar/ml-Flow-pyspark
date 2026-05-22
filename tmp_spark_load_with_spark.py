import os
import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark

os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
os.environ.setdefault('PYSPARK_DRIVER_PYTHON', sys.executable)

sp = SparkSession.builder.master('local[1]').appName('tmpSparkLoadTest').getOrCreate()
print('Spark context ready', sp)
model_path = 'mlruns/2/3e294f8be4e746cf9b4b43042081a49a/artifacts/model'
print('model path', os.path.abspath(model_path))
mdl = mlflow.spark.load_model(model_path)
print('loaded spark model', type(mdl))
sp.stop()
