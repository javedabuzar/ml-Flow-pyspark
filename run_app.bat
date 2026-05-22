@echo off
echo Starting FloodGuard AI API...
call d:\ml_flow_venv\Scripts\activate
set "JAVA_HOME=d:\ml flow abuzar"
set "HADOOP_HOME=C:\hardoop"
set "PYSPARK_PYTHON=d:\ml_flow_venv\Scripts\python.exe"
set "PYSPARK_DRIVER_PYTHON=d:\ml_flow_venv\Scripts\python.exe"
set "PYTHONFAULTHANDLER=1"
set "PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%PATH%"

uvicorn app:app --host 0.0.0.0 --port 8000
pause
