"""
predictor.py
============
Forecasting Engine.

Loads the trained LSTM model + scaler and produces multi-horizon power
consumption predictions (next 15 / 30 / 45 / 60 minutes) from the most
recent 24-hour window of readings.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocessing import build_latest_window, inverse_scale, load_scaler
from src.utils import HORIZON_LABELS, get_settings

TARGET_COL = "Global_active_power"


class EnergyForecaster:
    """Loads model artifacts once and serves multi-horizon forecasts."""

    def __init__(self, model_path: str | None = None, scaler_path: str | None = None):
        settings = get_settings()
        self.settings = settings
        self.model_path = model_path or settings.model_path
        self.scaler_path = scaler_path or settings.scaler_path
        self.model = None
        self.scaler = None

    def is_ready(self) -> bool:
        """Whether trained artifacts exist on disk."""
        return os.path.exists(self.model_path) and os.path.exists(self.scaler_path)

    def load(self) -> None:
        """Lazily load the Keras model and scaler into memory."""
        if self.model is None:
            self.model = keras.models.load_model(self.model_path)
        if self.scaler is None:
            self.scaler = load_scaler(self.scaler_path)

    def predict(self, df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
        """
        Predict next 15/30/45/60-minute power consumption from the latest
        `sequence_length` readings in `df`.

        Returns:
            {
              "current_usage": float,
              "prediction_15": float,
              "prediction_30": float,
              "prediction_45": float,
              "prediction_60": float,
            }
        """
        self.load()

        seq_len = self.settings.sequence_length
        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} historical readings to forecast, "
                f"got {len(df)}."
            )

        window = build_latest_window(df, self.scaler, sequence_length=seq_len, target_col=target_col)
        scaled_preds = self.model.predict(window, verbose=0).flatten()
        preds = inverse_scale(scaled_preds, self.scaler)
        preds = np.clip(preds, 0, None)

        current_usage = float(df[target_col].iloc[-1])

        result = {"current_usage": current_usage}
        for step, horizon in zip(self.settings.forecast_horizons, preds):
            label = HORIZON_LABELS.get(step, f"{step * 15} min")
            key = f"prediction_{label.split()[0]}"
            result[key] = float(horizon)

        return result

    def predict_series(self, df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
        """
        Return a small DataFrame with one row per forecast horizon, useful
        for plotting a forecast curve (current point + 4 future points).
        """
        result = self.predict(df, target_col=target_col)
        last_time = df["Datetime"].iloc[-1]

        rows = [{"Datetime": last_time, "Power": result["current_usage"], "Type": "Current"}]
        for step in self.settings.forecast_horizons:
            label = HORIZON_LABELS.get(step, f"{step * 15} min")
            key = f"prediction_{label.split()[0]}"
            rows.append(
                {
                    "Datetime": last_time + pd.Timedelta(minutes=step * 15),
                    "Power": result[key],
                    "Type": "Forecast",
                }
            )
        return pd.DataFrame(rows)
