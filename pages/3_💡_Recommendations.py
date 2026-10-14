"""Recommendations page — personalized energy-saving tips & alerts."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import hourly_profile
from src.data_loader import load_data
from src.predictor import EnergyForecaster
from src.recommendations import generate_alerts, generate_recommendations
from src.utils import get_settings

st.set_page_config(page_title="Recommendations | Smart Home Energy", page_icon="💡", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def _load(data_path: str, freq: str):
    return load_data(data_path, freq=freq)


@st.cache_resource(show_spinner=False)
def _forecaster():
    return EnergyForecaster()


def main():
    settings = get_settings()
    st.title("💡 Energy-Saving Recommendations")

    df = _load(settings.data_path, settings.resample_freq)
    forecaster = _forecaster()

    avg_power = float(df["Global_active_power"].mean())
    hourly = hourly_profile(df)
    peak_hour = int(hourly.loc[hourly["AvgPower"].idxmax(), "Hour"])

    if forecaster.is_ready() and len(df) >= settings.sequence_length:
        prediction = forecaster.predict(df)

        st.subheader("⚠️ Alerts")
        for alert in generate_alerts(prediction, avg_power, settings.high_consumption_threshold_pct):
            if alert["level"] == "danger":
                st.error(alert["message"])
            elif alert["level"] == "warning":
                st.warning(alert["message"])
            else:
                st.success(alert["message"])

        st.markdown("---")
        st.subheader("✅ Personalized Recommendations")
        for rec in generate_recommendations(prediction, avg_power, peak_hour):
            st.markdown(f"- {rec}")
    else:
        st.warning(
            "No trained model found yet — showing general recommendations based on "
            "historical usage patterns only."
        )
        for rec in generate_recommendations({"current_usage": avg_power}, avg_power, peak_hour):
            st.markdown(f"- {rec}")


main()
