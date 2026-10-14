"""
app.py
======
Smart Home Energy Forecasting using LSTM — Streamlit entry point (Dashboard page).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analytics import daily_consumption, hourly_profile, summary_kpis, trend_analysis
from src.data_loader import load_data
from src.predictor import EnergyForecaster
from src.recommendations import generate_alerts
from src.utils import BIG_CARD_CSS, format_currency, format_power, get_settings, render_forecast_cards

st.set_page_config(
    page_title="Smart Home Energy Forecasting",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(BIG_CARD_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading & cleaning household power data...")
def _load_clean_data(data_path: str, freq: str) -> pd.DataFrame:
    return load_data(data_path, freq=freq)


@st.cache_resource(show_spinner=False)
def _get_forecaster() -> EnergyForecaster:
    return EnergyForecaster()


def render_sidebar(settings) -> pd.DataFrame:
    """Sidebar: dataset source selector + global navigation info."""
    st.sidebar.title("🏠 Smart Home Energy")
    st.sidebar.caption("LSTM-powered electricity consumption forecasting")

    uploaded = st.sidebar.file_uploader(
        "Upload household power CSV (optional)", type=["csv"]
    )

    if uploaded is not None:
        tmp_path = os.path.join("data", "_uploaded.csv")
        os.makedirs("data", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        df = _load_clean_data(tmp_path, settings.resample_freq)
        st.sidebar.success(f"Loaded uploaded file: {len(df):,} records")
    else:
        df = _load_clean_data(settings.data_path, settings.resample_freq)
        st.sidebar.info(f"Using dataset: {len(df):,} records")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pages**\n"
        "- 📊 Dashboard (this page)\n"
        "- 📈 Forecast\n"
        "- 📉 Analytics\n"
        "- 💡 Recommendations\n"
        "- 🧪 Model Performance"
    )
    return df


def main() -> None:
    settings = get_settings()
    df = render_sidebar(settings)

    st.title("🏠 Smart Home Energy Forecasting — Dashboard")
    st.caption(
        "Monitor historical household electricity usage and preview the latest LSTM forecast."
    )

    forecaster = _get_forecaster()
    predicted_available = forecaster.is_ready() and len(df) >= settings.sequence_length

    st.subheader("⚡ Current Usage & Predicted Consumption")
    if predicted_available:
        prediction = forecaster.predict(df)
        render_forecast_cards(st, prediction)

        st.markdown("")
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
    else:
        prediction = None
        st.warning(
            "No trained model found yet. Run `python -m src.train_model` "
            "to train the LSTM forecaster, then reload this page."
        )

    st.markdown("---")

    kpis = summary_kpis(df, cost_per_kwh=settings.energy_cost_per_kwh)
    st.subheader("📊 Key Energy Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Usage", f"{kpis['avg_daily_kwh']:.1f} kWh")
    col2.metric("Peak Usage", format_power(kpis["peak_usage"]))
    col3.metric("Monthly Estimate", f"{kpis['monthly_estimate_kwh']:.0f} kWh")
    col4.metric(
        "Projected Monthly Cost",
        format_currency(kpis["projected_monthly_cost"], settings.currency_symbol),
    )

    st.markdown("---")
    st.subheader("📈 Recent Consumption Trend (last 7 days)")
    recent = df.tail(settings.sequence_length * 7)
    trend = trend_analysis(recent)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=trend["Datetime"], y=trend["Global_active_power"], name="Power (kW)", line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(x=trend["Datetime"], y=trend["RollingAvg"], name="24h Rolling Avg", line=dict(color="#ff7f0e", dash="dash"))
    )
    if predicted_available:
        series = forecaster.predict_series(df)
        forecast_points = series[series["Type"] == "Forecast"]
        fig.add_trace(
            go.Scatter(
                x=forecast_points["Datetime"], y=forecast_points["Power"],
                name="Forecast", mode="lines+markers", line=dict(color="#d62728", dash="dot"),
            )
        )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kW")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Daily Consumption (kWh)")
    daily = daily_consumption(df)
    fig_daily = go.Figure(go.Bar(x=daily["Datetime"], y=daily["kWh"], marker_color="#2ca02c"))
    fig_daily.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kWh")
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("🕒 Average Usage by Hour of Day")
    hourly = hourly_profile(df)
    fig_hourly = go.Figure(go.Bar(x=hourly["Hour"], y=hourly["AvgPower"], marker_color="#9467bd"))
    fig_hourly.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour", yaxis_title="Avg kW")
    st.plotly_chart(fig_hourly, use_container_width=True)


if __name__ == "__main__":
    main()
