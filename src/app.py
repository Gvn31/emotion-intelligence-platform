"""Streamlit emotion prediction dashboard with Prometheus metrics."""

import time

import streamlit as st
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

st.set_page_config(
    page_title="Emotion Intelligence",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def initialize_metrics():
    """Create process-wide metrics and expose /metrics on port 8000."""
    metrics = {
        "requests": Counter(
            "prediction_requests_total",
            "Total submitted emotion prediction requests",
        ),
        "successes": Counter(
            "predictions_total",
            "Total successful emotion predictions",
        ),
        "failures": Counter(
            "failed_predictions_total",
            "Total failed or invalid emotion prediction requests",
        ),
        "latency": Histogram(
            "prediction_latency_seconds",
            "Time spent producing an emotion prediction",
        ),
        "model_loaded": Gauge(
            "model_loaded",
            "Whether the validated emotion model loaded successfully",
        ),
        "model_health": Gauge(
            "model_health",
            "Whether the latest emotion inference succeeded",
        ),
        "model_info": Info(
            "model_info",
            "Validated emotion model identity",
        ),
        "confidence": Histogram(
            "prediction_confidence",
            "Prediction confidence as a ratio from 0 to 1",
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
        ),
        "emotions": Counter(
            "emotion_predictions_total",
            "Successful predictions by emotion",
            labelnames=("emotion",),
        ),
    }
    metrics["model_loaded"].set(0)
    metrics["model_health"].set(0)
    start_http_server(port=8000, addr="0.0.0.0")
    return metrics


metrics = initialize_metrics()

# predict.py loads the validated model during import. If that succeeds, it is ready.
from predict import predict_emotion

metrics["model_loaded"].set(1)
metrics["model_health"].set(1)
metrics["model_info"].info({"name": "emotion-classifier", "alias": "validated"})

st.title("Emotion Intelligence")
st.write("Enter text to predict its emotion.")

with st.form("prediction_form"):
    text = st.text_area("Text", placeholder="Type or paste text here…")
    submitted = st.form_submit_button("Predict emotion")

if submitted:
    metrics["requests"].inc()
    cleaned_text = text.strip()

    if not cleaned_text:
        metrics["failures"].inc()
        st.warning("Enter some text before requesting a prediction.")
    else:
        started_at = time.perf_counter()
        try:
            result = predict_emotion(cleaned_text)
            emotion = result["emotion"]
            confidence = float(result["confidence"])

            metrics["successes"].inc()
            metrics["emotions"].labels(emotion=emotion).inc()
            # predict.py returns percent; store confidence as a 0–1 ratio.
            metrics["confidence"].observe(max(0.0, min(confidence / 100.0, 1.0)))
            metrics["model_health"].set(1)

            st.subheader(f"Prediction: {emotion}")
            st.metric("Confidence", f"{confidence:.2f}%")

        except Exception as error:
            metrics["failures"].inc()
            metrics["model_health"].set(0)
            st.error(f"Prediction failed: {error}")
        finally:
            metrics["latency"].observe(time.perf_counter() - started_at)