FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 mlflow==2.13.0

# Crear directorio con permisos correctos
RUN mkdir -p /mlflow/artifacts && chmod -R 777 /mlflow

WORKDIR /mlflow

EXPOSE 5001

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5001"]
