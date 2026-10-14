"""
train_model.py
==============
Production LSTM training pipeline for Smart Home Energy Forecasting.

Architecture:
    Input -> LSTM(64, return_sequences=True) -> Dropout(0.2)
          -> LSTM(32) -> Dropout(0.2)
          -> Dense(16, relu) -> Dense(n_horizons)  [multi-horizon output]

Training features:
    - EarlyStopping (restores best weights)
    - ModelCheckpoint (persists best model to disk)
    - Validation split
    - Metrics: MAE, RMSE, MAPE, R2 (computed on the held-out test set)

Usage (CLI):
    python -m src.train_model
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

from src.data_loader import load_data
from src.preprocessing import inverse_scale, prepare_training_data
from src.utils import get_settings


def build_lstm_model(input_shape: tuple, n_outputs: int) -> keras.Model:
    """Construct the production LSTM architecture described above."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(n_outputs, activation="linear"),
        ],
        name="smart_home_lstm_forecaster",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """EarlyStopping + ModelCheckpoint callbacks for robust training."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler) -> dict:
    """Compute MAE, RMSE, MAPE, and R2 on the (inverse-scaled) test set."""
    y_pred = model.predict(X_test, verbose=0)

    y_test_flat = inverse_scale(y_test.flatten(), scaler)
    y_pred_flat = inverse_scale(y_pred.flatten(), scaler)

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = float(np.sqrt(np.mean((y_test_flat - y_pred_flat) ** 2)))
    # Avoid division-by-zero issues in MAPE for near-zero consumption readings.
    mape = mean_absolute_percentage_error(
        np.clip(y_test_flat, 1e-3, None), y_pred_flat
    ) * 100
    r2 = r2_score(y_test_flat, y_pred_flat)

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


def train(
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.1,
    data_path: str | None = None,
    verbose: int = 1,
) -> dict:
    """Run the full training pipeline and persist model + scaler + metrics."""
    settings = get_settings()
    data_path = data_path or settings.data_path

    df = load_data(data_path, freq=settings.resample_freq)
    prepared = prepare_training_data(
        df,
        sequence_length=settings.sequence_length,
        horizons=settings.forecast_horizons,
        scaler_path=settings.scaler_path,
    )

    model = build_lstm_model(
        input_shape=(settings.sequence_length, 1),
        n_outputs=len(settings.forecast_horizons),
    )

    history = model.fit(
        prepared["X_train"],
        prepared["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(settings.model_path),
        verbose=verbose,
    )

    # Ensure the final best model is saved (ModelCheckpoint already does this,
    # but we save again in case training finished without triggering it).
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    model.save(settings.model_path)

    metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"], prepared["scaler"])

    metrics_path = os.path.join(os.path.dirname(settings.model_path), "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print("Training complete. Metrics:", metrics)

    return {"model": model, "history": history.history, "metrics": metrics}


if __name__ == "__main__":
    train()
