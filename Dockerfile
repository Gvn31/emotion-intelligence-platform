FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    mlflow==2.22.1 \
    transformers \
    sentencepiece \
    joblib \
    scikit-learn \
    prometheus-client

COPY src/ ./src/
COPY models/emotion_model/ ./models/emotion_model/
COPY models/label_encoder.pkl ./models/label_encoder.pkl

EXPOSE 8051 8000

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8051", "--server.headless=true"]