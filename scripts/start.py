"""Train if needed, then start the FastAPI server.

Render.com injects a PORT environment variable — we bind uvicorn to it.
Set SKIP_TRAIN=1 to skip training (use pre-built model from models/).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "spark_pipeline"


def main() -> None:
    os.chdir(ROOT)

    skip = os.getenv("SKIP_TRAIN", "").lower() in ("1", "true", "yes")
    if not MODEL_PATH.exists() and not skip:
        print("No model found — training with PySpark (set TRAIN_MAX_ROWS for speed)...")
        env = os.environ.copy()
        env.setdefault("TRAIN_MAX_ROWS", "30000")
        subprocess.check_call([sys.executable, str(ROOT / "src" / "train.py")], env=env)
    elif not MODEL_PATH.exists() and skip:
        print("WARNING: SKIP_TRAIN=1 but no model found at models/spark_pipeline.")
        print("The API will attempt to load from MLflow on startup.")

    # Render sets PORT; fall back to API_PORT or 8000
    host = os.getenv("API_HOST", "0.0.0.0")
    port = os.getenv("PORT") or os.getenv("API_PORT", "8000")

    print(f"Starting uvicorn on {host}:{port}")
    subprocess.check_call(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", host, "--port", str(port)],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
