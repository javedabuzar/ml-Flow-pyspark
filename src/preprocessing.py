import os
import yaml
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Load params
params = yaml.safe_load(open("params.yaml"))

def preprocessing():
    spark = SparkSession.builder \
        .appName(f"{params['spark']['app_name']}_Preprocessing") \
        .master(params['spark']['master']) \
        .getOrCreate()

    # Load raw parquet data
    df = spark.read.parquet(params["data"]["raw_path"])
    
    # Notebook logic: clean data by dropping nulls in target
    target_col = "FloodProbability"
    df = df.na.drop(subset=[target_col])

    # Identify feature columns
    # In the notebook, all columns except ID and target are used
    input_cols = [c for c in df.columns if c not in ["id", target_col]]

    stages = []
    encoded_features = []

    # StringIndexer and OneHotEncoder for each feature (as per notebook)
    for col_name in input_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=col_name + "_indexed", outputCol=col_name + "_vec", handleInvalid="keep")
        stages.append(indexer)
        stages.append(encoder)
        encoded_features.append(col_name + "_vec")

    # VectorAssembler
    assembler = VectorAssembler(inputCols=encoded_features, outputCol="features")
    stages.append(assembler)

    # We create a partial pipeline for preprocessing
    preprocessing_pipeline = Pipeline(stages=stages)
    
    print("Fitting preprocessing pipeline...")
    preprocessing_model = preprocessing_pipeline.fit(df)
    df_transformed = preprocessing_model.transform(df)

    # Select features and label
    final_df = df_transformed.select("features", df[target_col].alias("label"))

    # Train-test split
    train_df, test_df = final_df.randomSplit(
        [1 - params["data"]["test_size"], params["data"]["test_size"]], 
        seed=params["data"]["random_state"]
    )

    # Save processed data
    os.makedirs(params["data"]["processed_path"], exist_ok=True)
    train_df.write.mode("overwrite").parquet(os.path.join(params["data"]["processed_path"], "train.parquet"))
    test_df.write.mode("overwrite").parquet(os.path.join(params["data"]["processed_path"], "test.parquet"))

    # Save preprocessing model
    os.makedirs("src/models", exist_ok=True)
    preprocessing_model.save("src/models/preprocessing_pipeline_model")

    print(f"Processed data saved to {params['data']['processed_path']}")
    print("Preprocessing pipeline model saved to src/models/preprocessing_pipeline_model")
    
    spark.stop()

if __name__ == "__main__":
    preprocessing()