FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    pytest==8.2.0 \
    pytest-json-report==1.5.0

COPY entrypoint.sh /entrypoint.sh
COPY run_tests.py .
COPY test_data.py .
COPY test_model.py .
COPY test_api.py .
COPY test_performance_drift.py .
COPY performance_drift_detector.py .

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
