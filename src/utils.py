"""
utils.py
=========
Shared configuration, constants, and helper utilities used across the
Smart Home Energy Forecasting application.

Loads runtime configuration from a `.env` file (via python-dotenv) so the
same codebase can run locally or on Streamlit Community Cloud by simply
changing environment variables / Streamlit secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file if present. On Streamlit Cloud,
# these can instead be supplied via st.secrets / the app settings UI.
load_dotenv()


def _get_env(name: str, default: str) -> str:
    """Return an environment variable, falling back to a default value."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Central, typed application configuration."""

    data_path: str
    model_path: str
    scaler_path: str
    sequence_length: int
    resample_freq: str
    forecast_horizons: tuple
    energy_cost_per_kwh: float
    currency_symbol: str
    high_consumption_threshold_pct: float


def get_settings() -> Settings:
    """Build and return application settings from environment variables."""
    horizons_raw = _get_env("FORECAST_HORIZONS", "1,2,3,4")
    horizons = tuple(int(h) for h in horizons_raw.split(",") if h.strip())

    return Settings(
        data_path=_get_env("DATA_PATH", "data/household_power.csv"),
        model_path=_get_env("MODEL_PATH", "models/lstm_model.keras"),
        scaler_path=_get_env("SCALER_PATH", "models/scaler.joblib"),
        sequence_length=int(_get_env("SEQUENCE_LENGTH", "96")),
        resample_freq=_get_env("RESAMPLE_FREQ", "15min"),
        forecast_horizons=horizons,
        energy_cost_per_kwh=float(_get_env("ENERGY_COST_PER_KWH", "0.15")),
        currency_symbol=_get_env("CURRENCY_SYMBOL", "$"),
        high_consumption_threshold_pct=float(
            _get_env("HIGH_CONSUMPTION_THRESHOLD_PCT", "25")
        ),
    )


# Human-readable labels for each forecast horizon step (step * 15 minutes).
HORIZON_LABELS = {1: "15 min", 2: "30 min", 3: "45 min", 4: "60 min"}


def minutes_for_step(step: int) -> int:
    """Convert a horizon step (in 15-minute increments) to minutes."""
    return step * 15


def format_power(value: float) -> str:
    """Format a power value (kW) for display."""
    return f"{value:.2f} kW"


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a monetary value for display."""
    return f"{symbol}{value:,.2f}"


# CSS for the large Current Usage / Forecast stat cards shared across pages.
BIG_CARD_CSS = """
<style>
.big-card {
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    color: white;
    height: 100%;
}
.big-card .label {
    font-size: 0.85rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
}
.big-card .value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.1;
}
.big-card .delta {
    font-size: 0.9rem;
    margin-top: 0.3rem;
    opacity: 0.95;
}
.card-current { background: linear-gradient(135deg, #1f4e8c, #2b6cb0); }
.card-up { background: linear-gradient(135deg, #b7282e, #e05353); }
.card-down { background: linear-gradient(135deg, #1a7a4c, #2eb872); }
.card-flat { background: linear-gradient(135deg, #5a5a5a, #808080); }
</style>
"""


def render_big_card(label: str, value: str, css_class: str, delta: str | None = None) -> str:
    """Build the HTML for one large stat card (Current Usage / Forecast)."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (
        f'<div class="big-card {css_class}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"{delta_html}"
        f"</div>"
    )


def render_forecast_cards(st_module, prediction: dict) -> None:
    """Render the Current Usage card + 4 forecast cards in one clear row."""
    current = prediction["current_usage"]
    horizons = [
        ("prediction_15", "Next 15 min"),
        ("prediction_30", "Next 30 min"),
        ("prediction_45", "Next 45 min"),
        ("prediction_60", "Next 60 min"),
    ]

    cols = st_module.columns(5)
    cols[0].markdown(
        render_big_card("Current Usage", format_power(current), "card-current"),
        unsafe_allow_html=True,
    )

    for col, (key, label) in zip(cols[1:], horizons):
        value = prediction[key]
        diff = value - current
        if diff > 0.05:
            css_class, arrow = "card-up", "▲"
        elif diff < -0.05:
            css_class, arrow = "card-down", "▼"
        else:
            css_class, arrow = "card-flat", "►"
        delta = f"{arrow} {diff:+.2f} kW vs current"
        col.markdown(
            render_big_card(label, format_power(value), css_class, delta),
            unsafe_allow_html=True,
        )

