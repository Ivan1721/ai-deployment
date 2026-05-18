FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    pytest==8.2.0 \
    pytest-json-report==1.5.0 \
    httpx==0.27.0 \
    great-expectations==0.18.15

COPY . .

# El exit code refleja si todos los tests pasan
# 0 = todos pasan (se puede promover)
# 1 = algún test falla (se bloquea la promoción)
CMD ["python", "run_tests.py"]
