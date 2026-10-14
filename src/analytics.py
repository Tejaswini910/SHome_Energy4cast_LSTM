"""
analytics.py
============
Energy Analytics Module.

Computes descriptive statistics and business-facing KPIs from the clean
household power consumption dataframe:
    - Current consumption
    - Average daily usage
    - Peak / lowest usage
    - Estimated monthly consumption
    - Projected energy cost
    - Daily / weekly / monthly aggregates and peak-hour analysis
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "Global_active_power"
READINGS_PER_HOUR = 4  # 15-minute interval data


def _energy_kwh(power_series: pd.Series, interval_hours: float = 0.25) -> float:
    """Convert a series of average-power (kW) readings into total energy (kWh)."""
    return float(power_series.sum() * interval_hours)


def total_consumption_kwh(df: pd.DataFrame, target_col: str = TARGET_COL) -> float:
    return _energy_kwh(df[target_col])


def daily_consumption(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Total energy (kWh) consumed per calendar day."""
    d = df.set_index("Datetime")[target_col].resample("D").sum() * 0.25
    return d.rename("kWh").reset_index()


def weekly_consumption(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Total energy (kWh) consumed per week."""
    w = df.set_index("Datetime")[target_col].resample("W").sum() * 0.25
    return w.rename("kWh").reset_index()


def monthly_consumption(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Total energy (kWh) consumed per month."""
    m = df.set_index("Datetime")[target_col].resample("ME").sum() * 0.25
    return m.rename("kWh").reset_index()


def hourly_profile(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Average power consumption by hour-of-day (0-23) -> peak usage hours."""
    tmp = df.copy()
    tmp["Hour"] = tmp["Datetime"].dt.hour
    profile = tmp.groupby("Hour")[target_col].mean().rename("AvgPower").reset_index()
    return profile


def weekday_hour_heatmap(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Pivot table of average power by weekday x hour, for a peak-usage heatmap."""
    tmp = df.copy()
    tmp["Hour"] = tmp["Datetime"].dt.hour
    tmp["Weekday"] = tmp["Datetime"].dt.day_name()
    pivot = tmp.pivot_table(index="Weekday", columns="Hour", values=target_col, aggfunc="mean")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in weekday_order if d in pivot.index])
    return pivot


def trend_analysis(df: pd.DataFrame, target_col: str = TARGET_COL, window: int = 96) -> pd.DataFrame:
    """Rolling mean trend line (default: 24h rolling average at 15-min resolution)."""
    tmp = df[["Datetime", target_col]].copy()
    tmp["RollingAvg"] = tmp[target_col].rolling(window=window, min_periods=1).mean()
    return tmp


def summary_kpis(
    df: pd.DataFrame,
    cost_per_kwh: float = 0.15,
    target_col: str = TARGET_COL,
) -> dict:
    """
    Compute the headline KPI dict used across the dashboard:
        current, avg_daily, peak, lowest, monthly_estimate, projected_cost
    """
    current = float(df[target_col].iloc[-1])
    daily = daily_consumption(df, target_col)
    avg_daily_kwh = float(daily["kWh"].mean()) if not daily.empty else 0.0
    peak = float(df[target_col].max())
    lowest = float(df[target_col].min())

    # Project a 30-day month from the observed average daily consumption.
    monthly_estimate_kwh = avg_daily_kwh * 30
    projected_monthly_cost = monthly_estimate_kwh * cost_per_kwh

    return {
        "current_consumption": current,
        "avg_daily_kwh": avg_daily_kwh,
        "peak_usage": peak,
        "lowest_usage": lowest,
        "monthly_estimate_kwh": monthly_estimate_kwh,
        "projected_monthly_cost": projected_monthly_cost,
        "total_consumption_kwh": total_consumption_kwh(df, target_col),
    }
