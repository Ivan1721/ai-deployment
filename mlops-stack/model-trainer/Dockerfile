FROM python:3.11-slim

WORKDIR /app

# Dependencias pesadas primero → Docker las cachea aunque cambie train.py
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    mlflow==2.13.0

COPY train.py .

CMD ["python", "train.py"]
