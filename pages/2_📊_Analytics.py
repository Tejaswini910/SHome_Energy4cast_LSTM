"""Analytics page — EDA: daily/weekly/monthly trends, peak hours, distribution, heatmap."""

from __future__ import annotations

import os
import sys

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import (
    daily_consumption,
    hourly_profile,
    monthly_consumption,
    summary_kpis,
    trend_analysis,
    weekday_hour_heatmap,
    weekly_consumption,
)
from src.data_loader import load_data
from src.utils import format_currency, format_power, get_settings

st.set_page_config(page_title="Analytics | Smart Home Energy", page_icon="📉", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def _load(data_path: str, freq: str):
    return load_data(data_path, freq=freq)


def main():
    settings = get_settings()
    st.title("📉 Analytics & Exploratory Data Analysis")

    df = _load(settings.data_path, settings.resample_freq)
    kpis = summary_kpis(df, cost_per_kwh=settings.energy_cost_per_kwh)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Consumption", f"{kpis['total_consumption_kwh']:.0f} kWh")
    c2.metric("Avg Daily Usage", f"{kpis['avg_daily_kwh']:.1f} kWh")
    c3.metric("Peak Usage", format_power(kpis["peak_usage"]))
    c4.metric("Lowest Usage", format_power(kpis["lowest_usage"]))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Daily / Weekly / Monthly", "Peak Usage Hours", "Distribution", "Trend", "Heatmap"]
    )

    with tab1:
        daily = daily_consumption(df)
        weekly = weekly_consumption(df)
        monthly = monthly_consumption(df)

        st.plotly_chart(px.bar(daily, x="Datetime", y="kWh", title="Daily Consumption (kWh)"),
                         use_container_width=True)
        st.plotly_chart(px.bar(weekly, x="Datetime", y="kWh", title="Weekly Consumption (kWh)"),
                         use_container_width=True)
        st.plotly_chart(px.bar(monthly, x="Datetime", y="kWh", title="Monthly Consumption (kWh)"),
                         use_container_width=True)

    with tab2:
        hourly = hourly_profile(df)
        fig = px.bar(hourly, x="Hour", y="AvgPower", title="Average Power by Hour of Day (Peak Usage Hours)")
        peak_hour = int(hourly.loc[hourly["AvgPower"].idxmax(), "Hour"])
        st.info(f"🕒 Historical peak usage hour: **{peak_hour}:00**")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.histogram(df, x="Global_active_power", nbins=50, title="Consumption Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        trend = trend_analysis(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["Datetime"], y=trend["Global_active_power"], name="Power", opacity=0.4))
        fig.add_trace(go.Scatter(x=trend["Datetime"], y=trend["RollingAvg"], name="24h Rolling Avg"))
        fig.update_layout(title="Long-Term Trend Analysis", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        pivot = weekday_hour_heatmap(df)
        fig = px.imshow(
            pivot,
            labels=dict(x="Hour of Day", y="Weekday", color="Avg kW"),
            aspect="auto",
            title="Peak Usage Heatmap (Weekday x Hour)",
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption(
        f"Projected monthly cost at {format_currency(settings.energy_cost_per_kwh, settings.currency_symbol)}/kWh: "
        f"**{format_currency(kpis['projected_monthly_cost'], settings.currency_symbol)}**"
    )


main()
