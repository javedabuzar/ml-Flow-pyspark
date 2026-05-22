@echo off
echo [1/4] Activating Virtual Environment...
call d:\ml_flow_venv\Scripts\activate
set JAVA_HOME=d:\ml flow abuzar
set HADOOP_HOME=C:\hardoop
set PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%PATH%

echo [2/4] Installing Dependencies One-by-One...
set PIP_CMD=pip install --cache-dir d:\pip_cache --default-timeout=1000

echo Installing MLflow, Kagglehub, PyYAML, Dotenv, Dagshub...
%PIP_CMD% mlflow kagglehub pyyaml python-dotenv dagshub

echo [CRITICAL] Installing PySpark (450MB+)... 
%PIP_CMD% pyspark

echo [3/4] Running ML Pipeline...
python src/data-ingestion.py
if %errorlevel% neq 0 (echo Data Ingestion Failed! && pause && exit /b %errorlevel%)

python src/preprocessing.py
if %errorlevel% neq 0 (echo Preprocessing Failed! && pause && exit /b %errorlevel%)

python src/train.py
if %errorlevel% neq 0 (echo Training Failed! && pause && exit /b %errorlevel%)

python src/best_model.py
if %errorlevel% neq 0 (echo Model Registration Failed! && pause && exit /b %errorlevel%)

echo [4/4] Starting MLflow UI...
mlflow ui --backend-store-uri sqlite:///mlflow.db
pause
