from pyspark.sql import SparkSession
import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")

try:
    spark = SparkSession.builder.master("local[1]").appName("Test").getOrCreate()
    df = spark.createDataFrame([{"test": 1}])
    result = df.collect()
    print(f"Spark Success: {result}")
    spark.stop()
except Exception as e:
    print(f"Spark Failed: {e}")
