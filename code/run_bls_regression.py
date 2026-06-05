"""
Run BLS public-data regression with industry-specific wage coefficients:

    log(openings_it) = α_i + β_i · log(wage_it) + ε_it

Estimated as a single pooled OLS with industry fixed effects and
industry × log(wage) interactions (no global intercept), which is
algebraically equivalent to separate OLS per industry but gives one
consistent set of residuals.

Usage:
    python code/run_bls_regression.py                    # all industries in bls_panel.csv → bls10_*
    python code/run_bls_regression.py --prefix bls       # same data, different output prefix
    python code/run_bls_regression.py --keep "Manufacturing" "Leisure and Hospitality" ...

Run from project root. Requires outputs/bls_panel.csv (fetch_bls.py first).
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"

CUTOFF = pd.Timestamp("2022-11-01")

FOUR_INDUSTRIES = [
    "Manufacturing",
    "Trade, Transportation, Utilities",
    "Professional and Business Services",
    "Leisure and Hospitality",
]


def grp_stats(s: pd.Series) -> dict:
    jb_s, jb_p = stats.jarque_bera(s)
    return {
        "n":      int(len(s)),
        "mean":   round(float(s.mean()),        6),
        "std":    round(float(s.std()),          6),
        "skew":   round(float(s.skew()),         6),
        "kurt":   round(float(s.kurtosis()),     6),
        "median": round(float(s.median()),       6),
        "p10":    round(float(s.quantile(.10)),  6),
        "p25":    round(float(s.quantile(.25)),  6),
        "p75":    round(float(s.quantile(.75)),  6),
        "p90":    round(float(s.quantile(.90)),  6),
        "jb_p":   round(float(jb_p),            8),
    }


def run_regression(panel: pd.DataFrame, out_prefix: str):
    n_ind = panel["industry"].nunique()
    print(f"\nFitting: {len(panel):,} obs, {n_ind} industries  →  {out_prefix}_*")

    # ── interaction model: separate α_i and β_i per industry ─────────────────
    # C(industry) gives one dummy per industry (no global intercept ⟹ full set)
    # C(industry):log_wage gives the industry-specific wage slopes
    formula = "log_openings ~ C(industry) + C(industry):log_wage - 1"
    model   = smf.ols(formula, data=panel).fit(cov_type="HC3")

    panel = panel.copy()
    panel["residual"] = model.resid.values
    panel["fitted"]   = model.fittedvalues.values

    # ── per-industry coefficients ─────────────────────────────────────────────
    industries_out = {}
    for ind in sorted(panel["industry"].unique()):
        int_key  = f"C(industry)[{ind}]"
        wage_key = f"C(industry)[{ind}]:log_wage"

        grp = panel[panel["industry"] == ind]
        ss_res = (grp["residual"] ** 2).sum()
        ss_tot = ((grp["log_openings"] - grp["log_openings"].mean()) ** 2).sum()
        r2_ind = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        industries_out[ind] = {
            "n":          int(len(grp)),
            "const_coef": round(float(model.params.get(int_key,  np.nan)), 6),
            "w_coef":     round(float(model.params.get(wage_key, np.nan)), 6),
            "w_se":       round(float(model.bse.get(wage_key,    np.nan)), 6),
            "w_p":        round(float(model.pvalues.get(wage_key,np.nan)), 6),
            "r2":         round(r2_ind, 6),
        }

    # ── pooled residual stats ─────────────────────────────────────────────────
    resid      = panel["residual"]
    jb_stat, jb_p = stats.jarque_bera(resid)

    pre_r  = panel.loc[panel["date"] <  CUTOFF, "residual"]
    post_r = panel.loc[panel["date"] >= CUTOFF, "residual"]
    ks_stat, ks_p = stats.ks_2samp(pre_r, post_r)

    results = {
        "n":            int(model.nobs),
        "n_industries": n_ind,
        "date_min":     str(panel["date"].min().date()),
        "date_max":     str(panel["date"].max().date()),
        "cutoff":       str(CUTOFF.date()),
        "r2_overall":   round(float(model.rsquared), 6),
        "industries":   industries_out,
        "resid_std":    round(float(resid.std()),      6),
        "resid_skew":   round(float(resid.skew()),     6),
        "resid_kurt":   round(float(resid.kurtosis()), 6),
        "resid_med":    round(float(resid.median()),   6),
        "jb_stat":      round(float(jb_stat),          4),
        "jb_p":         round(float(jb_p),             8),
        "ks_stat":      round(float(ks_stat),           6),
        "ks_p":         round(float(ks_p),              8),
        "pre":          grp_stats(pre_r),
        "post":         grp_stats(post_r),
    }

    # ── save ─────────────────────────────────────────────────────────────────
    res_path   = OUTPUTS / f"{out_prefix}_regression_results.json"
    panel_path = OUTPUTS / f"{out_prefix}_panel_with_residuals.csv"

    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    panel.to_csv(panel_path, index=False)

    print(f"Saved → {res_path}")
    print(f"Saved → {panel_path}")

    # ── per-industry summary ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Industry':<42}  {'β':>8}  {'SE':>7}  {'p':>6}  {'R²':>6}")
    print(f"{'─'*65}")
    for ind, d in industries_out.items():
        stars = ("***" if d["w_p"] < .01 else
                 "**"  if d["w_p"] < .05 else
                 "*"   if d["w_p"] < .10 else "")
        print(f"  {ind:<40}  {d['w_coef']:+8.4f}{stars:<3}  "
              f"{d['w_se']:7.4f}  {d['w_p']:6.3f}  {d['r2']:6.4f}")
    print(f"{'─'*65}")
    print(f"Overall R² (interaction model) = {results['r2_overall']:.4f}")
    print(f"KS test pre vs post Nov 2022: stat={ks_stat:.4f}  p={ks_p:.4f}")
    print(f"Residual skew={results['resid_skew']:.3f}  JB p={results['jb_p']:.2e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", nargs="*", default=None,
                        help="Filter to these industries (default: all)")
    parser.add_argument("--prefix", default=None,
                        help="Output file prefix (default: bls for 4-ind, bls10 for all)")
    args = parser.parse_args()

    panel_path = OUTPUTS / "bls_panel.csv"
    if not panel_path.exists():
        sys.exit("outputs/bls_panel.csv not found — run code/fetch_bls.py first")

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    print(f"Loaded panel: {len(panel):,} obs, {panel['industry'].nunique()} industries")

    # ── run for 4-industry benchmark ─────────────────────────────────────────
    panel4 = panel[panel["industry"].isin(FOUR_INDUSTRIES)].copy()
    run_regression(panel4, "bls")

    # ── run for all industries ────────────────────────────────────────────────
    run_regression(panel, "bls10")
