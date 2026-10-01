# SignalWatch — Marketing Anomaly Detection Agent

An AI agent that monitors marketing KPIs, statistically detects significant anomalies, drills down to the likely driving segment, and generates a plain-English alert with a recommended action — combining statistical detection with LLM-powered diagnosis rather than relying on either alone.

## The core idea

Most "AI agent" anomaly detection demos let an LLM eyeball a time series and guess whether something looks unusual. That's unreliable, and unnecessary — real statistical methods already do this well.

This project splits the problem instead:

```
Monitor  →  Detect  →  Diagnose  →  Explain / Recommend
(BigQuery)  (stats)    (stats)      (LLM — Gemini)
```

- **Detect**: day-of-week rolling baselines + robust z-scores (median/MAD rather than mean/std, so a single outlier week like Black Friday doesn't distort what counts as "normal")
- **Diagnose**: segment breakdowns ranked by *contribution to the overall change*, not by raw segment size — this avoids a classic Simpson's-paradox-style trap where the biggest segment looks like "the driver" just because it's big
- **Explain / Recommend**: an LLM (Gemini) turns the already-computed numbers into a short natural-language alert. It never performs detection itself and is explicitly instructed not to invent or estimate any figure beyond what it's given — severity itself is computed in plain Python, not left to the model, so it's reproducible

## Data

Built on Google's [GA4 BigQuery public sample dataset](https://console.cloud.google.com/bigquery(cloud:bigquery?project=bigquery-public-data&d=ga4_obfuscated_sample_ecommerce)) — real, obfuscated event data from the Google Merchandise Store (Nov 2020 – Jan 2021), accessible for free via BigQuery's sandbox tier. No real company or personal data is used anywhere in this project.

## Project structure

```
├── sql/
│   ├── 01_daily_kpis.sql          # Daily KPI extraction (traffic, conversion, revenue, AOV)
│   └── 02_segment_breakdown.sql   # Date x device x channel breakdown for diagnosis
├── src/
│   ├── kpi_baseline.py            # Detection: rolling + day-of-week baselines, robust z-scores
│   ├── diagnosis.py                # Segment drill-down, ranks contributors to the deviation
│   ├── llm_alert.py                 # Gemini-powered explain/recommend layer
│   └── report.py                    # Builds a styled, shareable HTML report
├── data/                          # Generated outputs (gitignored — see below)
├── marketing_anomaly_agent.ipynb # Main notebook: runs the full pipeline end to end
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
gcloud auth application-default login   # for BigQuery access
```

You'll also need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) — no billing account required. Set it as an environment variable before launching Jupyter:

```bash
export GEMINI_API_KEY="your-key-here"
jupyter lab
```

Open `marketing_anomaly_agent.ipynb`, set `PROJECT_ID` to your own GCP project ID in the config cell, and run top to bottom.

## Example output

```
🟠 Medium: Revenue increased on 2020-11-23

- Revenue reached 8482.0000 against a baseline of 4789.0000 (+77.1% change).
- Mobile / (none) revenue rose to 1684.0000 vs baseline 333.0000, contributing +1351.0000.
- Desktop / organic revenue rose to 1130.0000 vs baseline 98.0000, contributing +1032.0000.

Likely drivers: mobile / (none), desktop / organic

Recommendation: Investigate the underlying traffic and conversion data for
mobile / (none) and desktop / organic to identify what caused the sudden
revenue spike on this date.
```

## Notable design decisions / lessons learned

A few things that came up while tuning this that are worth knowing if you adapt it:

- **Median/MAD over mean/std** for the day-of-week baseline — with only ~13 weekly samples, a mean-based approach gets skewed badly by one outlier week (Black Friday, Christmas).
- **A floor under the MAD** — a day-of-week bucket can look artificially "tight" by chance, especially for low-variance KPIs, which otherwise turns a normal fluctuation into an absurd z-score. The floor is set relative to that KPI's overall variability.
- **A rolling window, not an expanding one** — using *all* history for the baseline backfires if a KPI's underlying level genuinely shifts partway through (e.g. a tracking/definition change); the baseline needs to be able to "forget" stale history within a few weeks.
- **A `reset_dates` mechanism** for known, confirmed level shifts (e.g. a tracking change), so the detector doesn't spend weeks re-flagging a shift it's already seen.
- **Diagnosis ranks by contribution, not size** — an additive KPI's biggest segment will naturally show the largest raw swings; ranking by contribution-to-delta (and, for ratio KPIs, weighting by traffic/purchase share) avoids a large-but-stable segment looking like "the driver."

## Tech stack

Google BigQuery · Google Gemini (free tier) · Python (pandas) — the detection and diagnosis logic operates entirely on DataFrames, so it's portable to other data sources (Adobe Analytics, Postgres, Snowflake, etc.) by swapping out the extraction layer.

## License

MIT
