"""
llm_alert.py

The "Explain + Recommend" step of Monitor -> Detect -> Diagnose -> Recommend.

Takes the STRUCTURED facts already computed by kpi_baseline.py and
diagnosis.py (the anomaly's KPI, its % deviation, and the ranked segment
contributions) and asks Gemini to turn them into a short, plain-English
alert with a recommendation.

Deliberately NOT doing here: any numeric judgment. Severity, direction, and
the ranked driver list are all computed in Python beforehand and simply
handed to the model as facts. The LLM's only job is turning already-correct
numbers into readable prose — it is not asked to detect the anomaly, decide
if it's significant, or invent numbers that weren't given to it. This
keeps the model from ever being in a position to hallucinate a stat.

Usage:
    from llm_alert import get_client, generate_alert

    client = get_client(project_id=PROJECT_ID, location="us-central1")
    alert_text = generate_alert(
        client,
        kpi="conversion_rate",
        anomaly_date="2020-11-27",
        current_value=0.0089,
        baseline_value=0.0142,
        pct_change=-0.373,
        severity="High",
        ranked_segments=ranked,   # the DataFrame from diagnosis.diagnose()
    )
    print(alert_text)
"""

import pandas as pd

DEFAULT_MODEL = "gemini-3.6-flash"


def get_client(api_key: str = None):
    """
    Creates a Gemini client via the free Gemini Developer API
    (ai.google.dev) — no GCP billing account required, just an API key.

    If `api_key` isn't passed explicitly, this looks for it in the
    GEMINI_API_KEY environment variable.
    """
    import os
    from google import genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Either pass get_client(api_key='...') "
            "directly, or set the GEMINI_API_KEY environment variable "
            "before starting Jupyter."
        )
    return genai.Client(api_key=api_key)


def severity_from_z(z_score: float) -> str:
    """
    Deterministic, non-LLM severity classification from the z-score already
    computed by kpi_baseline.py. Kept as plain Python so severity is always
    reproducible and never something the model could vary between runs.
    """
    z = abs(z_score)
    if z >= 6:
        return "High"
    elif z >= 4:
        return "Medium"
    else:
        return "Low"


def _segment_lines(ranked_segments: pd.DataFrame, kpi: str) -> str:
    """Formats the ranked segment table as plain text facts for the prompt."""
    lines = []
    non_metric_cols = [
        c for c in ranked_segments.columns
        if c not in ("users", kpi, f"{kpi}_baseline", "contribution")
    ]
    for _, row in ranked_segments.iterrows():
        segment_desc = " / ".join(str(row[c]) for c in non_metric_cols)
        direction = "down" if row["contribution"] < 0 else "up"
        lines.append(
            f"- {segment_desc}: {kpi} {direction} "
            f"({row[kpi]:.4f} vs baseline {row[f'{kpi}_baseline']:.4f}), "
            f"contribution to overall change: {row['contribution']:+.4f}"
        )
    return "\n".join(lines)


def build_prompt(
    kpi: str,
    anomaly_date,
    current_value: float,
    baseline_value: float,
    pct_change: float,
    severity: str,
    ranked_segments: pd.DataFrame,
) -> str:
    date_str = pd.Timestamp(anomaly_date).date()
    segment_facts = _segment_lines(ranked_segments, kpi)

    return f"""You are a marketing analytics assistant. Below are FACTS that have
already been computed by a statistical detection pipeline. Do not invent,
estimate, or add any numbers beyond what is given here — only use these
exact figures.

KPI: {kpi}
Date: {date_str}
Current value: {current_value:.4f}
Baseline (expected) value: {baseline_value:.4f}
Percent change vs baseline: {pct_change:+.1%}
Severity: {severity}

Top contributing segments (ranked by contribution to the overall change,
not by raw traffic size):
{segment_facts}

Write a short alert in this exact structure:
1. A one-line headline with a severity emoji (🔴 High, 🟠 Medium, 🟡 Low)
   naming the KPI and direction of change.
2. 2-3 bullet points citing the specific numbers above (do not round
   differently than given, do not add other metrics not listed here).
3. A "Likely drivers" section listing the top 1-2 segments from the facts
   above, in plain English (e.g. "mobile / organic").
4. A single "Recommendation" sentence — a concrete next investigative step,
   not a decision to change budget or spend. Do not recommend specific
   dollar amounts or actions beyond what is grounded in the facts above.

Keep the whole thing under 120 words. Do not use markdown headers, just
plain lines."""


def generate_alert(
    client,
    kpi: str,
    anomaly_date,
    current_value: float,
    baseline_value: float,
    pct_change: float,
    severity: str,
    ranked_segments: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    retry_delay_seconds: float = 15.0,
) -> str:
    """
    Same as before, but retries automatically on transient server errors
    (e.g. 503 UNAVAILABLE / "high demand") rather than failing the whole
    run over what's usually a temporary hiccup on Google's end.
    """
    import time

    prompt = build_prompt(
        kpi, anomaly_date, current_value, baseline_value, pct_change,
        severity, ranked_segments,
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            is_last_attempt = attempt == max_retries
            if is_last_attempt:
                raise
            print(f"  (attempt {attempt} failed: {e} — retrying in {retry_delay_seconds}s)")
            time.sleep(retry_delay_seconds)

    raise last_error  # pragma: no cover — unreachable, satisfies linters
