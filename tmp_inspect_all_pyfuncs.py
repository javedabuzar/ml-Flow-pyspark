import glob
import mlflow
from pathlib import Path

paths = sorted(glob.glob('mlruns/1/models/*/artifacts'), key=lambda p: Path(p).stat().st_mtime, reverse=True)
for p in paths:
    print('\n===', p)
    try:
        mdl = mlflow.pyfunc.load_model(p)
        inner = mdl._model_impl
        print('type:', type(inner))
        print('sklearn type:', type(inner.sklearn_model))
        print('n_features_in_', getattr(inner.sklearn_model, 'n_features_in_', 'NA'))
        print('has feature_names_in_', hasattr(inner.sklearn_model, 'feature_names_in_'))
        if hasattr(inner.sklearn_model, 'feature_names_in_'):
            print('feature_names_in', inner.sklearn_model.feature_names_in_)
    except Exception as e:
        print('load failed', e)
