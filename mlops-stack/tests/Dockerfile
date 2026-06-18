FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    pyyaml==6.0.1

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    pytest==8.2.0 \
    pytest-json-report==1.5.0

# Backends de almacenamiento opcionales (se activan según el YAML de config)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    dvc==3.50.2 \
    dvc-azure==3.1.0 \
    dvc-gdrive==3.0.1 \
    azure-storage-blob==12.20.0 \
    azure-identity==1.17.1

COPY mlops_common ./mlops_common
COPY configs ./configs
COPY tests/entrypoint.sh /entrypoint.sh
COPY tests/run_tests.py .
COPY tests/test_data.py .
COPY tests/test_model.py .
COPY tests/test_api.py .
COPY tests/test_performance_drift.py .
COPY tests/performance_drift_detector.py .

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
