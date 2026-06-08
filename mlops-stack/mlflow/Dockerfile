FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 mlflow==2.13.0

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5001

# El entrypoint crea /mlflow/artifacts y fija permisos DESPUÉS
# de que Docker monte el volumen — esto es lo que fallaba antes
ENTRYPOINT ["/entrypoint.sh"]
