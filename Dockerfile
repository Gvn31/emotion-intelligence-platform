FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install CPU-only PyTorch separately from the other packages.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    mlflow==2.22.1 \
    transformers \
    sentencepiece \
    joblib \
    scikit-learn

COPY src/ ./src/
COPY models/emotion_model/ ./models/emotion_model/
COPY models/label_encoder.pkl ./models/label_encoder.pkl

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]