FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias científicas primero (capa cacheada)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0 \
    scikit-learn==1.4.2 \
    pyyaml==6.0.1

# MLFlow aparte (grande)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

COPY mlops_common ./mlops_common
COPY configs ./configs
COPY drift-detector/detector.py .
COPY drift-detector/retrain_trigger.py .
COPY drift-detector/performance_drift_detector.py .

CMD ["python", "detector.py"]
