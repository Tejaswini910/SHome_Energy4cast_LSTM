"""Forecast page — multi-horizon LSTM predictions with a forecast curve."""

from __future__ import annotations

import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.predictor import EnergyForecaster
from src.recommendations import generate_alerts
from src.utils import BIG_CARD_CSS, get_settings, render_forecast_cards

st.set_page_config(page_title="Forecast | Smart Home Energy", page_icon="📈", layout="wide")
st.markdown(BIG_CARD_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def _load(data_path: str, freq: str):
    return load_data(data_path, freq=freq)


@st.cache_resource(show_spinner=False)
def _forecaster():
    return EnergyForecaster()


def main():
    settings = get_settings()
    st.title("📈 Forecast — Next 15 / 30 / 45 / 60 Minutes")

    df = _load(settings.data_path, settings.resample_freq)
    forecaster = _forecaster()

    if not forecaster.is_ready():
        st.warning(
            "No trained model found. Train the model first by running "
            "`python -m src.train_model` from the project root, then reload this page."
        )
        return

    if len(df) < settings.sequence_length:
        st.error(f"Need at least {settings.sequence_length} historical readings to forecast.")
        return

    prediction = forecaster.predict(df)

    st.subheader("Current vs Predicted Consumption")
    render_forecast_cards(st, prediction)

    st.markdown("---")
    st.subheader("Forecast Curve")
    series = forecaster.predict_series(df)
    history = df.tail(settings.sequence_length).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Datetime"], y=history["Global_active_power"],
                              name="Last 24h (actual)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=series["Datetime"], y=series["Power"],
                              name="Forecast", mode="lines+markers", line=dict(color="#d62728", dash="dash")))
    fig.update_layout(height=460, yaxis_title="kW", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Alerts")
    alerts = generate_alerts(
        prediction,
        avg_daily_power=df["Global_active_power"].mean(),
        threshold_pct=settings.high_consumption_threshold_pct,
    )
    for alert in alerts:
        if alert["level"] == "danger":
            st.error(alert["message"])
        elif alert["level"] == "warning":
            st.warning(alert["message"])
        else:
            st.success(alert["message"])


main()
