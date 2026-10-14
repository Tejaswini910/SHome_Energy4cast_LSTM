"""Model Performance page — training metrics and actual vs predicted charts."""

from __future__ import annotations

import json
import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.predictor import EnergyForecaster
from src.preprocessing import inverse_scale, prepare_training_data
from src.utils import get_settings

st.set_page_config(page_title="Model Performance | Smart Home Energy", page_icon="🧪", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def _load(data_path: str, freq: str):
    return load_data(data_path, freq=freq)


@st.cache_resource(show_spinner=False)
def _forecaster():
    return EnergyForecaster()


def main():
    settings = get_settings()
    st.title("🧪 Model Performance & Evaluation")

    forecaster = _forecaster()
    if not forecaster.is_ready():
        st.warning(
            "No trained model found. Run `python -m src.train_model` to train the LSTM "
            "forecaster, then reload this page."
        )
        return

    forecaster.load()
    df = _load(settings.data_path, settings.resample_freq)

    metrics_path = os.path.join(os.path.dirname(settings.model_path), "metrics.json")
    st.subheader("📊 Evaluation Metrics")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        cols = st.columns(4)
        cols[0].metric("MAE", f"{metrics.get('MAE', 0):.3f} kW")
        cols[1].metric("RMSE", f"{metrics.get('RMSE', 0):.3f} kW")
        cols[2].metric("MAPE", f"{metrics.get('MAPE', 0):.2f} %")
        cols[3].metric("R² Score", f"{metrics.get('R2', 0):.3f}")
    else:
        st.info("No saved metrics.json found. Re-run training to generate evaluation metrics.")

    st.markdown("---")
    st.subheader("Actual vs Predicted (held-out test set)")

    prepared = prepare_training_data(
        df,
        sequence_length=settings.sequence_length,
        horizons=settings.forecast_horizons,
        scaler_path=None,
    )
    # Reuse the persisted scaler (fit on full history) for inverse transforms,
    # ensuring consistent units with the trained model.
    scaler = forecaster.scaler
    y_pred_scaled = forecaster.model.predict(prepared["X_test"], verbose=0)

    # Focus on the 15-minute-ahead horizon (first output) for the chart.
    y_true = inverse_scale(prepared["y_test"][:, 0], scaler)
    y_pred = inverse_scale(y_pred_scaled[:, 0], scaler)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(y=y_pred, name="Predicted (15 min horizon)", line=dict(color="#d62728")))
    fig.update_layout(height=420, yaxis_title="kW", xaxis_title="Test sample index")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Error")
    error = y_true - y_pred
    fig_err = go.Figure(go.Scatter(y=error, mode="lines", name="Error (Actual - Predicted)", line=dict(color="#ff7f0e")))
    fig_err.update_layout(height=320, yaxis_title="Error (kW)", xaxis_title="Test sample index")
    st.plotly_chart(fig_err, use_container_width=True)


main()
