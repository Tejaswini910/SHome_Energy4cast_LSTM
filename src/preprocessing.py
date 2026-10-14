"""
preprocessing.py
================
Data Preprocessing module.

Responsible for:
    - Null handling (delegated to data_loader, re-validated here).
    - Feature scaling with MinMaxScaler (persisted via Joblib).
    - Sliding-window sequence creation for supervised time-series learning.
    - Train/test split that preserves temporal order (no shuffling).

Window scheme:
    Past 24 Hours (96 x 15-minute records) -> Predict -> next 15/30/45/60 min
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "Global_active_power"


def validate_no_nulls(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Raise if nulls remain in the target column after cleaning."""
    if df[target_col].isna().any():
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df


def fit_scaler(values: np.ndarray) -> MinMaxScaler:
    """Fit a MinMaxScaler on a 1D/2D array of values."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values.reshape(-1, 1))
    return scaler


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)


def scale_series(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Scale a 1D array of raw values using a fitted scaler."""
    return scaler.transform(values.reshape(-1, 1)).flatten()


def inverse_scale(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform scaled values back to the original kW units."""
    arr = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()


def create_sequences(
    scaled_values: np.ndarray,
    sequence_length: int = 96,
    horizons: tuple = (1, 2, 3, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for multi-horizon forecasting.

    For every index `i`, X contains the previous `sequence_length` readings
    and y contains the target values at `i + h` for each horizon step `h`
    in `horizons` (e.g. steps 1..4 correspond to 15/30/45/60 minutes ahead
    when the data is resampled to 15-minute intervals).

    Returns:
        X: shape (n_samples, sequence_length, 1)
        y: shape (n_samples, len(horizons))
    """
    max_horizon = max(horizons)
    n = len(scaled_values)
    X, y = [], []

    for i in range(sequence_length, n - max_horizon + 1):
        X.append(scaled_values[i - sequence_length:i])
        y.append([scaled_values[i + h - 1] for h in horizons])

    X = np.array(X, dtype=np.float32).reshape(-1, sequence_length, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_test_split_sequences(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/test split (no shuffling) so the test set always
    represents the most recent time period, avoiding data leakage.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 96,
    horizons: tuple = (1, 2, 3, 4),
    test_size: float = 0.2,
    target_col: str = TARGET_COL,
    scaler_path: str | None = None,
):
    """
    End-to-end preprocessing pipeline producing scaled, windowed
    train/test tensors ready for the LSTM model.

    Returns a dict with X_train, X_test, y_train, y_test, scaler.
    """
    df = validate_no_nulls(df, target_col)
    values = df[target_col].values.astype(np.float32)

    scaler = fit_scaler(values)
    if scaler_path:
        save_scaler(scaler, scaler_path)

    scaled = scale_series(values, scaler)
    X, y = create_sequences(scaled, sequence_length=sequence_length, horizons=horizons)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, test_size=test_size)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def build_latest_window(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    sequence_length: int = 96,
    target_col: str = TARGET_COL,
) -> np.ndarray:
    """
    Build the most recent scaled window (shape: 1, sequence_length, 1) used
    to forecast the next 15/30/45/60 minutes from "now".
    """
    values = df[target_col].values.astype(np.float32)[-sequence_length:]
    scaled = scale_series(values, scaler)
    return scaled.reshape(1, sequence_length, 1)
