FROM python:3.11-slim

WORKDIR /app

# sklearn/pandas/numpy first — cached if only train.py changes
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2

# system dependency required by LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# gradient-boosting libraries
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

COPY train.py .

CMD ["python", "train.py"]
