import mlflow
from pathlib import Path
p = Path('mlruns/2/3e294f8be4e746cf9b4b43042081a49a/artifacts/model')
print('path', p.exists(), p)
mdl = mlflow.spark.load_model(str(p))
print('loaded model type', type(mdl))
print('model class', mdl.__class__)
