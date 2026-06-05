"""
Fetch JOLTS job openings and CES average hourly earnings from FRED.

FRED (Federal Reserve Economic Data) mirrors BLS data and provides free
CSV downloads with no authentication. BLS bulk downloads are blocked for
programmatic access.

Industries matched (JOLTS x CES available on FRED):
  Manufacturing          JTS3000JOL  x  CES3000000003
  Trade/Transport/Util   JTS4000JOL  x  CES4142000003
  Prof & Business Svcs   JTS6000JOL  x  CES6000000003
  Leisure & Hospitality  JTS7000JOL  x  CES7000000003

Run from project root:
    python code/fetch_bls.py

Output:
    outputs/bls_panel.csv
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT    = Path(__file__).parent.parent
RAW_BLS = ROOT / "data" / "raw" / "bls"
OUTPUTS = ROOT / "outputs"
RAW_BLS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS  = {"User-Agent": "wage-as-signal-research/1.0 (academic)"}

START = "2012-01-01"
END   = "2024-12-01"

# ── series registry ───────────────────────────────────────────────────────────

JOLTS_SERIES = {
    "Manufacturing"                  : "JTS3000JOL",
    "Trade, Transportation, Utilities": "JTS4000JOL",
    "Professional and Business Services": "JTS6000JOL",
    "Leisure and Hospitality"        : "JTS7000JOL",
}

CES_SERIES = {
    "Manufacturing"                  : "CES3000000003",
    "Trade, Transportation, Utilities": "CES4142000003",
    "Professional and Business Services": "CES6000000003",
    "Leisure and Hospitality"        : "CES7000000003",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_fred(series_id: str, cache_path: Path) -> pd.Series:
    """Download a FRED series as a dated Series, caching the CSV locally."""
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
    df = df[(df.index >= START) & (df.index <= END)]
    return df


def load_all(series_dict: dict, subfolder: str) -> pd.DataFrame:
    folder = RAW_BLS / subfolder
    folder.mkdir(exist_ok=True)
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
    print("\n-- JOLTS job openings (FRED) --")
    jolts = load_all(JOLTS_SERIES, "jolts")
    jolts = jolts.rename(columns={"value": "openings_thousands"})
    print(f"  {len(jolts):,} obs, {jolts['industry'].nunique()} industries")

    print("\n-- CES average hourly earnings (FRED) --")
    ces = load_all(CES_SERIES, "ces")
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
    print(f"  date range: {panel['date'].min().date()} - {panel['date'].max().date()}")
    print(f"  pre-ChatGPT: {n_pre}   post-ChatGPT: {n_post}")
    for ind, g in panel.groupby("industry"):
        print(f"    {ind:40s}  {len(g)} obs  "
              f"wage ${g['avg_hourly_earnings'].mean():.2f}/hr  "
              f"openings {g['openings_thousands'].mean():.0f}k")

    out = OUTPUTS / "bls_panel.csv"
    panel.to_csv(out, index=False)
    print(f"\nSaved -> {out}  ({len(panel):,} rows)")
