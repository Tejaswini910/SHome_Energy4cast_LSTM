"""
recommendations.py
===================
Warning/Alert System + Recommendation Engine.

Generates dynamic, rule-based alerts and personalized energy-saving
recommendations from forecast results and historical KPIs.
"""

from __future__ import annotations


def generate_alerts(
    prediction: dict,
    avg_daily_power: float,
    threshold_pct: float = 25.0,
) -> list[dict]:
    """
    Build a list of alert dicts: {"level": "warning"/"info"/"danger", "message": str}

    Compares predicted future usage against the historical average power
    level and flags significant spikes.
    """
    alerts: list[dict] = []
    current = prediction.get("current_usage", 0.0)
    horizons = [
        ("prediction_15", "next 15 minutes"),
        ("prediction_30", "next 30 minutes"),
        ("prediction_45", "next 45 minutes"),
        ("prediction_60", "next 60 minutes"),
    ]

    if avg_daily_power <= 0:
        avg_daily_power = max(current, 0.01)

    for key, label in horizons:
        if key not in prediction:
            continue
        value = prediction[key]
        pct_over_avg = ((value - avg_daily_power) / avg_daily_power) * 100

        if pct_over_avg >= threshold_pct:
            alerts.append(
                {
                    "level": "danger",
                    "message": (
                        f"⚠️ Warning: Expected consumption spike in the {label} "
                        f"({value:.2f} kW, {pct_over_avg:.0f}% above average)."
                    ),
                }
            )
        elif pct_over_avg >= threshold_pct / 2:
            alerts.append(
                {
                    "level": "warning",
                    "message": (
                        f"Consumption in the {label} is trending "
                        f"{pct_over_avg:.0f}% above your average ({value:.2f} kW)."
                    ),
                }
            )

    if not alerts:
        alerts.append(
            {
                "level": "info",
                "message": "✅ Consumption levels are within the normal range for the forecast horizon.",
            }
        )
    else:
        alerts.append(
            {
                "level": "warning",
                "message": "Consider switching off high-power appliances to reduce the predicted spike.",
            }
        )

    return alerts


def generate_recommendations(prediction: dict, avg_daily_power: float, peak_hour: int | None = None) -> list[str]:
    """
    Produce a personalized list of energy-saving recommendations based on
    predicted consumption patterns.
    """
    recs: list[str] = []
    forecasted = [
        prediction.get(k)
        for k in ("prediction_15", "prediction_30", "prediction_45", "prediction_60")
        if prediction.get(k) is not None
    ]
    max_forecast = max(forecasted) if forecasted else prediction.get("current_usage", 0.0)

    if avg_daily_power <= 0:
        avg_daily_power = max(max_forecast, 0.01)

    if max_forecast > avg_daily_power * 1.25:
        recs.append("🌡️ Reduce air conditioning / heating usage during the predicted peak window.")
        recs.append("🔌 Switch non-essential high-power appliances to power-saving mode.")
        recs.append("⏱️ Shift usage of heavy appliances (dishwasher, dryer) to off-peak hours.")

    if peak_hour is not None:
        recs.append(
            f"🕒 Your historical peak usage hour is around {peak_hour}:00 — "
            "consider scheduling washing machines and water heaters outside this window."
        )

    recs.append("💡 Use smart power strips to eliminate standby/phantom loads.")
    recs.append("🧺 Run washing machines and dryers with full loads during off-peak hours.")
    recs.append("📉 Set your thermostat 1-2°C back during high-tariff periods to cut peak-hour costs.")

    # De-duplicate while preserving order.
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    return unique_recs
