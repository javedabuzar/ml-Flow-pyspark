FROM python:3.11-slim

# Install Java (required for PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV SPARK_MASTER=local[1]
ENV SPARK_DRIVER_MEMORY=512m
ENV SPARK_EXECUTOR_MEMORY=512m
ENV SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY params.yaml .
COPY mlflow.db .
COPY src/ src/
COPY static/ static/
COPY mlruns/ mlruns/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
