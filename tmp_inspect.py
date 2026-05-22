import pandas as pd
import mlflow
from pathlib import Path

p = Path('mlruns/1/models/m-fc547de9935e400e80925150a5b2b036/artifacts')
print('model path', p)
print('MLmodel exists', (p / 'MLmodel').exists())
print('files:', sorted([x.name for x in p.iterdir()]))
mdl = mlflow.pyfunc.load_model(str(p))
print('pyfunc type', type(mdl))
inner = mdl._model_impl
print('impl type', type(inner))
print('sklearn type', type(inner.sklearn_model))
print('has feature_names_in_', hasattr(inner.sklearn_model, 'feature_names_in_'))
if hasattr(inner.sklearn_model, 'feature_names_in_'):
    print('feature_names_in_', inner.sklearn_model.feature_names_in_)
print('n_features_in_', getattr(inner.sklearn_model, 'n_features_in_', 'NA'))
print('sklearn feature attrs:', [a for a in dir(inner.sklearn_model) if 'feature' in a.lower()])
print('sklearn state keys:', list(inner.sklearn_model.__dict__.keys()))
print('params', inner.sklearn_model.get_params())
