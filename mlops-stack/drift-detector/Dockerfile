FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias científicas primero (capa cacheada)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0 \
    scikit-learn==1.4.2

# MLFlow aparte (grande)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

COPY performance_drift_detector.py .
COPY detector.py .
COPY retrain_trigger.py .

CMD ["python", "detector.py"]
