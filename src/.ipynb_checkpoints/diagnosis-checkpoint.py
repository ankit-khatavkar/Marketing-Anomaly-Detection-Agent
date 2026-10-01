"""
diagnosis.py

The "diagnose" step of Monitor -> Detect -> Diagnose -> Recommend.

Given a day flagged as anomalous for a given KPI, this pulls the segment
breakdown (date x device x medium x source) and ranks segments by their
CONTRIBUTION to the overall change — not by raw size. This matters: a
segment can look like "the driver" simply because it's the biggest slice
of traffic, even if its own rate barely moved (Simpson's-paradox-style
trap). Ranking by contribution avoids that.

Two kinds of KPIs are handled differently:

  - Additive KPIs (revenue, total_users): a segment's contribution is just
    its own change, since these sum directly across segments.
      contribution_i = segment_value_i_today - segment_value_i_baseline

  - Ratio KPIs (conversion_rate): a segment's contribution is its share of
    traffic times how much its own rate moved (mix-adjusted). This is a
    first-order decomposition — it ignores the small cross term from
    traffic mix shifting at the same time the rate shifts, which is a
    reasonable simplification for a first cut.
      contribution_i = (segment_users_i_today / total_users_today)
                        * (segment_rate_i_today - segment_rate_i_baseline)

Usage:
    from diagnosis import run_query, diagnose

    segment_df = run_query("sql/02_segment_breakdown.sql", PROJECT_ID)
    segment_df["event_date"] = pd.to_datetime(segment_df["event_date"])

    diagnose(segment_df, kpi="conversion_rate",
             anomaly_date="2020-11-27", segment_cols=["device_category", "medium"])
"""

from pathlib import Path

import numpy as np
import pandas as pd

RATIO_KPIS = {
    "conversion_rate": "users",        # weighted by traffic (visitors)
    "avg_order_value": "purchasers",   # weighted by purchases, not visitors
}
ADDITIVE_KPIS = {"revenue", "total_users"}  # sums directly across segments

# KPIs the segment SQL doesn't currently support at all (no underlying
# columns to derive them from). Surfaced explicitly rather than silently
# giving a wrong answer.
UNSUPPORTED_KPIS = {"engagement_rate"}

# The daily KPI table (01_daily_kpis.sql) and the segment breakdown table
# (02_segment_breakdown.sql) were written independently and don't share
# identical column names for the same underlying thing — the daily table
# calls it "total_users", the segment table calls it "users". This maps
# a KPI name as it appears in the daily table's anomaly_driver_kpi column
# to the column name actually present in segment_df.
KPI_TO_SEGMENT_COL = {
    "total_users": "users",
}


def run_query(sql_path: str, project_id: str) -> pd.DataFrame:
    """Same helper as kpi_baseline.run_query — kept here too so this module
    can be used standalone."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    sql = Path(sql_path).read_text()
    return client.query(sql).to_dataframe()


def _segment_baseline(
    segment_df: pd.DataFrame,
    segment_cols: list,
    anomaly_date: pd.Timestamp,
    day_of_week: str,
    window: int,
    value_col: str,
) -> pd.DataFrame:
    """
    For each segment, computes its median `value_col` over the last `window`
    prior occurrences of the SAME weekday — mirrors the day-of-week baseline
    logic used for the top-level KPIs, applied per segment instead.
    """
    hist = segment_df[
        (segment_df["event_date"] < anomaly_date)
        & (segment_df["event_date"].dt.day_name() == day_of_week)
    ]

    baseline = (
        hist.sort_values("event_date")
        .groupby(segment_cols)[value_col]
        .apply(lambda s: s.tail(window).median())
        .reset_index()
        .rename(columns={value_col: f"{value_col}_baseline"})
    )
    return baseline


def diagnose(
    segment_df: pd.DataFrame,
    kpi: str,
    anomaly_date,
    segment_cols: list = ["device_category", "medium"],
    window: int = 5,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Returns the top_n segments ranked by |contribution| to the change in
    `kpi` on `anomaly_date`, largest contributor first.
    """
    anomaly_date = pd.Timestamp(anomaly_date)
    day_of_week = anomaly_date.day_name()

    if kpi in UNSUPPORTED_KPIS:
        raise ValueError(
            f"'{kpi}' can't be diagnosed with the current segment SQL — "
            "02_segment_breakdown.sql doesn't track engagement at the "
            "segment level. Add engaged-session columns there first."
        )

    # Translate to the segment table's column name if it differs
    # (e.g. "total_users" in the daily table -> "users" in segment_df),
    # but keep using the original `kpi` name for display purposes.
    segment_col = KPI_TO_SEGMENT_COL.get(kpi, kpi)

    segment_df = segment_df.copy()
    if kpi == "avg_order_value" and "avg_order_value" not in segment_df.columns:
        segment_df["avg_order_value"] = (
            segment_df["revenue"] / segment_df["purchasers"]
        )
    if segment_col != kpi and kpi not in segment_df.columns:
        segment_df[kpi] = segment_df[segment_col]

    today = segment_df[segment_df["event_date"] == anomaly_date].copy()
    if today.empty:
        raise ValueError(f"No segment data found for {anomaly_date.date()}")

    if kpi in ADDITIVE_KPIS:
        baseline = _segment_baseline(
            segment_df, segment_cols, anomaly_date, day_of_week, window, kpi
        )
        merged = today.merge(baseline, on=segment_cols, how="left")
        merged["contribution"] = merged[kpi] - merged[f"{kpi}_baseline"]

    elif kpi in RATIO_KPIS:
        weight_col = RATIO_KPIS[kpi]
        rate_baseline = _segment_baseline(
            segment_df, segment_cols, anomaly_date, day_of_week, window, kpi
        )
        merged = today.merge(rate_baseline, on=segment_cols, how="left")

        total_weight_today = today[weight_col].sum()
        merged["traffic_share"] = merged[weight_col] / total_weight_today
        merged["contribution"] = merged["traffic_share"] * (
            merged[kpi] - merged[f"{kpi}_baseline"]
        )

    else:
        raise ValueError(
            f"'{kpi}' isn't configured as additive or ratio — add it to "
            "ADDITIVE_KPIS or RATIO_KPIS in diagnosis.py"
        )

    merged["abs_contribution"] = merged["contribution"].abs()
    ranked = merged.sort_values("abs_contribution", ascending=False).head(top_n)

    cols = segment_cols + ["users", kpi, f"{kpi}_baseline", "contribution"]
    return ranked[cols].reset_index(drop=True)


def explain(ranked: pd.DataFrame, kpi: str, anomaly_date) -> str:
    """
    Turns the ranked segment table into a short plain-English summary —
    the input an LLM reasoning step would take to write the final alert.
    This function itself does NOT call an LLM; it just prepares the
    structured facts so the LLM step has something grounded to work from.
    """
    lines = [f"Anomaly in {kpi} on {pd.Timestamp(anomaly_date).date()}:"]
    for _, row in ranked.iterrows():
        segment_desc = " / ".join(str(row[c]) for c in ranked.columns[: len(ranked.columns) - 4])
        direction = "down" if row["contribution"] < 0 else "up"
        lines.append(
            f"  - {segment_desc}: {kpi} {direction} "
            f"({row[kpi]:.4f} vs baseline {row[f'{kpi}_baseline']:.4f}), "
            f"contribution {row['contribution']:+.4f}"
        )
    return "\n".join(lines)
