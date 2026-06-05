"""
Fetch JOLTS job openings (BLS API) and CES average hourly earnings (FRED)
for 10 private-sector industries, Jan 2012 – Dec 2024.

BLS API key required in .env as BLS_API_KEY.
FRED CSV endpoint requires no authentication.

Series ID format (BLS API JOLTS, 21 chars):
  JTS + industry(6) + area(5=00000) + state(2=00) + sizeclass(2=00) + JO + L

Run from project root:
    python code/fetch_bls.py

Output:
    outputs/bls_panel.csv
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()
BLS_KEY = os.getenv("BLS_API_KEY", "")
if not BLS_KEY:
    raise RuntimeError("BLS_API_KEY missing from .env")

ROOT    = Path(__file__).parent.parent
RAW_BLS = ROOT / "data" / "raw" / "bls"
OUTPUTS = ROOT / "outputs"
for p in (RAW_BLS / "jolts", RAW_BLS / "ces", OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

BLS_API  = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS  = {"User-Agent": "wage-as-signal-research/1.0 (academic)"}

START_YEAR = 2012
END_YEAR   = 2024
START_DATE = "2012-01-01"
END_DATE   = "2024-12-01"

# ── series registries ─────────────────────────────────────────────────────────
# JOLTS: JTS + 6-digit BLS industry code + 000000000 (area/state/sizeclass) + JOL
# Codes discovered via DBnomics BLS/jt dataset metadata.

JOLTS_SERIES = {
    "Mining and Logging":                  "JTS110099000000000JOL",
    "Construction":                        "JTS230000000000000JOL",
    "Manufacturing":                       "JTS300000000000000JOL",
    "Trade, Transportation, Utilities":    "JTS400000000000000JOL",
    "Information":                         "JTS510000000000000JOL",
    "Financial Activities":                "JTS510099000000000JOL",
    "Professional and Business Services":  "JTS540099000000000JOL",
    "Education and Health Services":       "JTS600000000000000JOL",
    "Leisure and Hospitality":             "JTS700000000000000JOL",
    "Other Services":                      "JTS810000000000000JOL",
}

# CES AHE: all available on FRED
CES_SERIES = {
    "Mining and Logging":                  "CES0600000003",
    "Construction":                        "CES2000000003",
    "Manufacturing":                       "CES3000000003",
    "Trade, Transportation, Utilities":    "CES4142000003",
    "Information":                         "CES5000000003",
    "Financial Activities":                "CES5500000003",
    "Professional and Business Services":  "CES6000000003",
    "Education and Health Services":       "CES6500000003",
    "Leisure and Hospitality":             "CES7000000003",
    "Other Services":                      "CES8000000003",
}


# ── BLS API fetch ─────────────────────────────────────────────────────────────

def fetch_bls_api(series_dict: dict) -> pd.DataFrame:
    """Pull all JOLTS series from BLS API v2 in batches of 50."""
    items = list(series_dict.items())
    frames = []

    for batch_start in range(0, len(items), 50):
        batch = items[batch_start : batch_start + 50]
        label_map = {sid: label for label, sid in batch}

        # BLS API allows 20-year span per call; split if needed
        for yr_start in range(START_YEAR, END_YEAR + 1, 20):
            yr_end = min(yr_start + 19, END_YEAR)
            payload = {
                "seriesid":       list(label_map.keys()),
                "registrationkey": BLS_KEY,
                "startyear":      str(yr_start),
                "endyear":        str(yr_end),
            }
            r = requests.post(BLS_API, json=payload, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()

            for msg in data.get("message", []):
                if "does not exist" in msg:
                    print(f"  [WARN] {msg}")

            for series in data.get("Results", {}).get("series", []):
                sid  = series["seriesID"]
                rows = series.get("data", [])
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                df["date"]     = pd.to_datetime(df["year"].astype(str) + "-" + df["period"].str[1:] + "-01")
                df["value"]    = pd.to_numeric(df["value"], errors="coerce")
                df["industry"] = label_map[sid]
                df["series_id"] = sid
                frames.append(df[["industry", "series_id", "date", "value"]])

            time.sleep(0.3)  # be polite

    if not frames:
        raise RuntimeError("No JOLTS data returned from BLS API")

    out = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["value"])
        .query("@START_DATE <= date <= @END_DATE")
        .drop_duplicates(["industry", "date"])
        .sort_values(["industry", "date"])
        .reset_index(drop=True)
    )
    return out


# ── FRED fetch ────────────────────────────────────────────────────────────────

def fetch_fred(series_id: str, cache_path: Path) -> pd.Series:
    if cache_path.exists():
        print(f"  [cached]  {cache_path.name}")
    else:
        url = FRED_CSV.format(sid=series_id)
        print(f"  [GET]     {url}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        cache_path.write_text(r.text, encoding="utf-8")

    df = pd.read_csv(cache_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col  = next(c for c in df.columns if "date" in c)
    value_col = next(c for c in df.columns if c != date_col)
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date")["value"]
    return df[(df.index >= START_DATE) & (df.index <= END_DATE)]


def load_ces(series_dict: dict) -> pd.DataFrame:
    folder = RAW_BLS / "ces"
    frames = []
    for industry, sid in series_dict.items():
        s = fetch_fred(sid, folder / f"{sid}.csv")
        df = s.reset_index()
        df.columns = ["date", "value"]
        df["industry"] = industry
        df["series_id"] = sid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n-- JOLTS job openings (BLS API) --")
    jolts = fetch_bls_api(JOLTS_SERIES)
    jolts = jolts.rename(columns={"value": "openings_thousands"})
    print(f"  {len(jolts):,} obs, {jolts['industry'].nunique()} industries")
    for ind, g in jolts.groupby("industry"):
        print(f"    {ind:45s}  {len(g)} obs")

    print("\n-- CES average hourly earnings (FRED) --")
    ces = load_ces(CES_SERIES)
    ces = ces.rename(columns={"value": "avg_hourly_earnings"})
    print(f"  {len(ces):,} obs, {ces['industry'].nunique()} industries")

    print("\n-- building panel --")
    panel = jolts.merge(
        ces[["industry", "date", "avg_hourly_earnings"]],
        on=["industry", "date"],
        how="inner",
    )

    panel = panel[panel["openings_thousands"] > 0].copy()
    panel["year"]         = panel["date"].dt.year
    panel["month"]        = panel["date"].dt.month
    panel["log_openings"] = np.log(panel["openings_thousands"])
    panel["log_wage"]     = np.log(panel["avg_hourly_earnings"])
    panel["post_chatgpt"] = panel["date"] >= pd.Timestamp("2022-11-01")

    panel = panel.sort_values(["industry", "date"]).reset_index(drop=True)

    n_pre  = (~panel["post_chatgpt"]).sum()
    n_post = panel["post_chatgpt"].sum()
    print(f"  rows: {len(panel):,}  industries: {panel['industry'].nunique()}")
    print(f"  date range: {panel['date'].min().date()} – {panel['date'].max().date()}")
    print(f"  pre-ChatGPT: {n_pre}   post-ChatGPT: {n_post}")
    for ind, g in panel.groupby("industry"):
        print(f"    {ind:45s}  {len(g)} obs  "
              f"wage ${g['avg_hourly_earnings'].mean():.2f}/hr  "
              f"openings {g['openings_thousands'].mean():.0f}k")

    out = OUTPUTS / "bls_panel.csv"
    panel.to_csv(out, index=False)
    print(f"\nSaved -> {out}  ({len(panel):,} rows)")
