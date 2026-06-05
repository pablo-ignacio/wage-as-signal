"""
Run BLS public-data regression:
    log(openings_it) = α + β·log(wage_it) + ε_it
where i = industry, t = month.

Pre/post split at November 2022 (ChatGPT launch).

Run from project root:
    python code/run_bls_regression.py

Requires outputs/bls_panel.csv (run fetch_bls.py first).

Output:
    outputs/bls_regression_results.json
    outputs/bls_panel_with_residuals.csv
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"

CUTOFF = pd.Timestamp("2022-11-01")


def grp_stats(s: pd.Series) -> dict:
    jb_s, jb_p = stats.jarque_bera(s)
    return {
        "n":      int(len(s)),
        "mean":   round(float(s.mean()),          6),
        "std":    round(float(s.std()),            6),
        "skew":   round(float(s.skew()),           6),
        "kurt":   round(float(s.kurtosis()),       6),
        "p10":    round(float(s.quantile(0.10)),   6),
        "p25":    round(float(s.quantile(0.25)),   6),
        "median": round(float(s.median()),         6),
        "p75":    round(float(s.quantile(0.75)),   6),
        "p90":    round(float(s.quantile(0.90)),   6),
        "jb_p":   round(float(jb_p),              8),
    }


if __name__ == "__main__":
    panel_path = OUTPUTS / "bls_panel.csv"
    if not panel_path.exists():
        sys.exit("outputs/bls_panel.csv not found — run code/fetch_bls.py first")

    panel = pd.read_csv(panel_path, parse_dates=["date"])

    if len(panel) == 0:
        sys.exit("Panel is empty — check fetch_bls.py output")

    print(f"Panel: {len(panel):,} obs, {panel['industry'].nunique()} industries")
    print(f"Date:  {panel['date'].min().date()} – {panel['date'].max().date()}")
    print(f"Industries: {sorted(panel['industry'].unique())}")

    # ── OLS ──────────────────────────────────────────────────────────────────
    y = panel["log_openings"]
    X = sm.add_constant(panel["log_wage"])
    model = sm.OLS(y, X).fit(cov_type="HC3")

    panel = panel.copy()
    panel["residual"] = model.resid.values
    panel["fitted"]   = model.fittedvalues.values

    resid = panel["residual"]
    jb_stat, jb_p = stats.jarque_bera(resid)

    pre_r  = panel.loc[panel["date"] <  CUTOFF, "residual"]
    post_r = panel.loc[panel["date"] >= CUTOFF, "residual"]

    if len(pre_r) < 2 or len(post_r) < 2:
        sys.exit(f"Not enough obs in one period: pre={len(pre_r)}, post={len(post_r)}")

    ks_stat, ks_p = stats.ks_2samp(pre_r, post_r)

    # ── results dict ─────────────────────────────────────────────────────────
    results = {
        "n":             int(model.nobs),
        "n_industries":  int(panel["industry"].nunique()),
        "date_min":      str(panel["date"].min().date()),
        "date_max":      str(panel["date"].max().date()),
        "cutoff":        str(CUTOFF.date()),
        "r2":            round(float(model.rsquared),           6),
        "const_coef":    round(float(model.params["const"]),    6),
        "const_se":      round(float(model.bse["const"]),       6),
        "const_p":       round(float(model.pvalues["const"]),   6),
        "w_coef":        round(float(model.params["log_wage"]), 6),
        "w_se":          round(float(model.bse["log_wage"]),    6),
        "w_p":           round(float(model.pvalues["log_wage"]),6),
        "resid_std":     round(float(resid.std()),              6),
        "resid_skew":    round(float(resid.skew()),             6),
        "resid_kurt":    round(float(resid.kurtosis()),         6),
        "resid_med":     round(float(resid.median()),           6),
        "resid_p10":     round(float(resid.quantile(0.10)),     6),
        "resid_p25":     round(float(resid.quantile(0.25)),     6),
        "resid_p75":     round(float(resid.quantile(0.75)),     6),
        "resid_p90":     round(float(resid.quantile(0.90)),     6),
        "jb_stat":       round(float(jb_stat),                  4),
        "jb_p":          round(float(jb_p),                     8),
        "ks_stat":       round(float(ks_stat),                   6),
        "ks_p":          round(float(ks_p),                      8),
        "pre":           grp_stats(pre_r),
        "post":          grp_stats(post_r),
    }

    # ── save ─────────────────────────────────────────────────────────────────
    res_path = OUTPUTS / "bls_regression_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {res_path}")

    panel_out = OUTPUTS / "bls_panel_with_residuals.csv"
    panel.to_csv(panel_out, index=False)
    print(f"Saved → {panel_out}")

    # ── summary ──────────────────────────────────────────────────────────────
    r = results
    print(f"\n{'─'*55}")
    print(f"log(openings) = α + β·log(wage)   [pooled OLS, HC3]")
    print(f"N = {r['n']:,}  ({r['n_industries']} industries × monthly)")
    print(f"β = {r['w_coef']:+.4f}  SE={r['w_se']:.4f}  p={r['w_p']:.3f}")
    print(f"α = {r['const_coef']:+.4f}  SE={r['const_se']:.4f}")
    print(f"R² = {r['r2']:.4f}")
    print(f"KS test (pre vs post Nov 2022): stat={ks_stat:.4f}  p={ks_p:.4f}")
    print(f"Residual skew={r['resid_skew']:.3f}  JB p={r['jb_p']:.2e}")
