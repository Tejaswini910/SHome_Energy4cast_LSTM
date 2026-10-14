"""
Generate a bundled sample dataset (data/household_power.csv) so the app
and training pipeline can run out-of-the-box without requiring the full
~20MB UCI archive to be downloaded first.

Usage:
    python scripts/generate_sample_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_synthetic_data  # noqa: E402

if __name__ == "__main__":
    df = generate_synthetic_data(start="2024-01-01", days=90, freq="15min")
    out_path = os.path.join("data", "household_power.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
