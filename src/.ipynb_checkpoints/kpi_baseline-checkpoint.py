"""
kpi_baseline.py

Importable module version of the baseline/anomaly logic — no argparse, no
__main__ block. Meant to be called cell-by-cell from a notebook so you can
inspect intermediate DataFrames as you iterate on thresholds and logic.

    from kpi_baseline import (
        run_query,
        add_rolling_baselines,
        add_day_of_week_baseline,
        flag_anomalies,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config — tweak these directly in a notebook cell if you want to experiment,
# e.g. `import kpi_baseline; kpi_baseline.Z_ALERT_THRESHOLD = 2.5`
# ---------------------------------------------------------------------------

KPI_COLUMNS = [
    "total_users",
    "engagement_rate",
    "conversion_rate",
    "revenue",
    "avg_order_value",
]

Z_ALERT_THRESHOLD = 3.0          # |robust z| above this -> anomaly
MAD_SCALE = 1.4826               # scales MAD to be comparable to std dev
                                  # under a normal distribution assumption


# ---------------------------------------------------------------------------
# Data pull
# ---------------------------------------------------------------------------

def run_query(sql_path: str, project_id: str) -> pd.DataFrame:
    """Runs the given SQL file against BigQuery and returns a DataFrame.

    Requires Application Default Credentials to be set up
    (`gcloud auth application-default login`) or
    GOOGLE_APPLICATION_CREDENTIALS pointing at a service account key.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    sql = Path(sql_path).read_text()
    return client.query(sql).to_dataframe()


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

def add_rolling_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 7-day and 30-day trailing rolling averages (excluding today)."""
    df = df.sort_values("event_date").reset_index(drop=True)

    for col in KPI_COLUMNS:
        # shift(1) so "today" is never included in its own baseline
        df[f"{col}_avg_7d"] = (
            df[col].shift(1).rolling(window=7, min_periods=3).mean()
        )
        df[f"{col}_avg_30d"] = (
            df[col].shift(1).rolling(window=30, min_periods=10).mean()
        )
        df[f"{col}_pct_vs_7d"] = (
            (df[col] - df[f"{col}_avg_7d"]) / df[f"{col}_avg_7d"]
        )
        df[f"{col}_pct_vs_30d"] = (
            (df[col] - df[f"{col}_avg_30d"]) / df[f"{col}_avg_30d"]
        )

    return df


def add_day_of_week_baseline(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Median + MAD baseline per weekday, computed from a ROLLING window of the
    last `window` prior occurrences of that weekday (not all history).

    Using all history (an expanding window) sounds more "data-rich," but it
    backfires if the metric's underlying level genuinely shifts partway
    through — the median stays anchored to the stale old level for the rest
    of the series, so every day after a real shift keeps looking anomalous
    even though nothing unusual is actually happening anymore. A rolling
    window lets the baseline adapt within a few weeks of a real level shift,
    while still being large enough (window=5) to average out single-week
    noise.
    """
    df = df.sort_values("event_date").reset_index(drop=True)

    for col in KPI_COLUMNS:
        med_col = f"{col}_dow_median"
        mad_col = f"{col}_dow_mad"
        z_col = f"{col}_robust_z"

        # Overall (whole-series) MAD for this KPI, used as a floor below.
        # A small rolling bucket can look artificially "tight" just by
        # chance, especially for low-variance KPIs like engagement_rate —
        # dividing by that tiny MAD then turns a normal fluctuation into a
        # z-score of 10+. Flooring the bucket MAD at a fraction of the
        # KPI's overall MAD prevents that blow-up.
        overall_med = df[col].median()
        overall_mad = (df[col] - overall_med).abs().median()
        mad_floor = overall_mad * 0.3  # tune this fraction if needed

        med_vals, mad_vals = [], []
        for _, row in df.iterrows():
            history = df[
                (df["day_of_week"] == row["day_of_week"])
                & (df["event_date"] < row["event_date"])
            ][col].tail(window)  # <-- only the most recent `window` occurrences

            if len(history) >= 3:
                med = history.median()
                mad = (history - med).abs().median()
                mad = max(mad, mad_floor)  # apply floor
            else:
                med, mad = np.nan, np.nan

            med_vals.append(med)
            mad_vals.append(mad)

        df[med_col] = med_vals
        df[mad_col] = mad_vals

        # robust z-score; guard against MAD == 0 (e.g. overall_mad also 0)
        mad_safe = df[mad_col].replace(0, np.nan) * MAD_SCALE
        df[z_col] = (df[col] - df[med_col]) / mad_safe

    return df


def flag_anomalies(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Flags a day as anomalous if any KPI's robust z-score exceeds threshold."""
    threshold = Z_ALERT_THRESHOLD if threshold is None else threshold
    z_cols = [f"{col}_robust_z" for col in KPI_COLUMNS]

    def worst_driver(row):
        vals = {c: row[c] for c in z_cols if pd.notna(row[c])}
        if not vals:
            return None, np.nan
        worst = max(vals, key=lambda c: abs(vals[c]))
        return worst.replace("_robust_z", ""), vals[worst]

    df = df.copy()
    df["is_anomaly"] = df[z_cols].abs().max(axis=1, skipna=True) >= threshold
    df[["anomaly_driver_kpi", "anomaly_driver_z"]] = df.apply(
        lambda r: pd.Series(worst_driver(r)), axis=1
    )
    return df


def build_baseline_table(df: pd.DataFrame, threshold: float = None, dow_window: int = 5) -> pd.DataFrame:
    """Convenience wrapper: runs all three baseline steps in sequence."""
    df = add_rolling_baselines(df)
    df = add_day_of_week_baseline(df, window=dow_window)
    df = flag_anomalies(df, threshold=threshold)
    return df
