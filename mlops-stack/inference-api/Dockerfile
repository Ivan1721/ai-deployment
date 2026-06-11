FROM python:3.11-slim

WORKDIR /app

# system dependency required by LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias pesadas en capas separadas
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2

# gradient-boosting libraries needed to deserialize trained models
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
