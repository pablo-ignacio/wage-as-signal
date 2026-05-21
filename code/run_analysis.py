"""
Runs the labor demand regression and saves results to outputs/.
Run from project root: python code/run_analysis.py

The Streamlit app reads from outputs/ so it works without the raw data.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(".")
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

OCC_VAR         = "role_k50"
MIN_OTHER_FIRMS = 2


def build_panel(df):
    d = df.dropna(subset=["rcid", OCC_VAR, "year", "salary"]).copy()
    d = d[d["salary"] > 0]
    d["rcid"] = d["rcid"].astype(int)
    d["year"] = d["year"].astype(int)

    firm_panel = (
        d.groupby(["rcid", OCC_VAR, "year"])
        .agg(posting_count=("job_id", "count"),
             firm_wage_sum=("salary", "sum"),
             firm_n_wages=("salary", "count"))
        .reset_index()
    )
    occ_totals = (
        d.groupby([OCC_VAR, "year"])
        .agg(occ_wage_sum=("salary", "sum"),
             occ_n_wages=("salary", "count"),
             occ_n_firms=("rcid", "nunique"))
        .reset_index()
    )
    p = firm_panel.merge(occ_totals, on=[OCC_VAR, "year"])
    loo_n          = p["occ_n_wages"] - p["firm_n_wages"]
    p["w_loo"]     = (p["occ_wage_sum"] - p["firm_wage_sum"]) / loo_n
    p["loo_n"]     = loo_n
    p["log_w_loo"] = np.log(p["w_loo"])
    p["log_count"] = np.log(p["posting_count"])
    p = p[(p["occ_n_firms"] - 1) >= MIN_OTHER_FIRMS]
    p = p[p["loo_n"] > 0].dropna(subset=["w_loo", "log_w_loo"])
    return p


def run_ols(panel, w_col):
    y = panel["log_count"]
    X = sm.add_constant(panel[w_col])
    return sm.OLS(y, X).fit(cov_type="HC3")


print("Loading data...")
raw = pd.read_csv(
    ROOT / "data/raw/revelio_academic_postings/academic_postings_unified_individual_academic.csv",
    low_memory=False
)
raw["year"]            = pd.to_datetime(raw["post_date"], errors="coerce").dt.year
raw["salary_observed"] = raw["salary_predicted"].astype(str).str.strip().str.lower() == "false"

print("Building panels...")
all_panel = build_panel(raw)
obs_panel = build_panel(raw[raw["salary_observed"]])

specs = {
    "(1) Levels — All wages"      : (all_panel, "w_loo"),
    "(2) Log wage — All wages"    : (all_panel, "log_w_loo"),
    "(3) Levels — Observed wages" : (obs_panel, "w_loo"),
    "(4) Log wage — Observed wages": (obs_panel, "log_w_loo"),
}

print("Running regressions...")
reg_results = {}
for label, (panel, w_col) in specs.items():
    m = run_ols(panel, w_col)
    jb_stat, jb_p = stats.jarque_bera(m.resid)
    reg_results[label] = {
        "w_col"       : w_col,
        "wage_sample" : "All wages" if "All" in label else "Observed wages",
        "wage_var"    : "Levels" if "Levels" in label else "Log",
        "n"           : int(m.nobs),
        "r2"          : round(m.rsquared, 6),
        "const_coef"  : round(m.params["const"], 6),
        "const_se"    : round(m.bse["const"], 6),
        "const_p"     : round(m.pvalues["const"], 6),
        "w_coef"      : round(m.params[w_col], 10),
        "w_se"        : round(m.bse[w_col], 10),
        "w_p"         : round(m.pvalues[w_col], 6),
        "resid_std"   : round(m.resid.std(), 6),
        "resid_skew"  : round(m.resid.skew(), 6),
        "resid_kurt"  : round(m.resid.kurtosis(), 6),
        "resid_p10"   : round(m.resid.quantile(0.10), 6),
        "resid_p25"   : round(m.resid.quantile(0.25), 6),
        "resid_med"   : round(m.resid.median(), 6),
        "resid_p75"   : round(m.resid.quantile(0.75), 6),
        "resid_p90"   : round(m.resid.quantile(0.90), 6),
        "jb_stat"     : round(jb_stat, 4),
        "jb_p"        : round(jb_p, 8),
    }

# Save regression summary
with open(OUTPUTS / "regression_results.json", "w") as f:
    json.dump(reg_results, f, indent=2)
print("Saved outputs/regression_results.json")

# Save panels with residuals (for scatter and residual plots in app)
for label, (panel, w_col) in specs.items():
    m = run_ols(panel, w_col)
    out = panel[["rcid", OCC_VAR, "year", "posting_count", "log_count", "w_loo", "log_w_loo"]].copy()
    out["residual"] = m.resid.values
    out["fitted"]   = m.fittedvalues.values
    slug = label.split("—")[0].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = OUTPUTS / f"panel_{slug}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path.name}")

# ── ChatGPT era comparison (pre/post October 2022) ─────────────────────────
# Panel is at year level: pre = year <= 2022, post = year >= 2023
CHATGPT_CUTOFF = 2022   # postings in year <= this are "pre"

chatgpt_results = {}
for label, (panel, w_col) in specs.items():
    m    = run_ols(panel, w_col)
    resid = pd.Series(m.resid.values, index=panel.index)
    pre  = resid[panel["year"] <= CHATGPT_CUTOFF]
    post = resid[panel["year"] >  CHATGPT_CUTOFF]

    ks_stat, ks_p = stats.ks_2samp(pre, post) if (len(pre) > 0 and len(post) > 0) else (None, None)

    def grp_stats(s):
        if len(s) == 0:
            return {}
        jb_s, jb_p = stats.jarque_bera(s)
        return {
            "n"       : int(len(s)),
            "mean"    : round(float(s.mean()), 6),
            "std"     : round(float(s.std()), 6),
            "skew"    : round(float(s.skew()), 6),
            "kurt"    : round(float(s.kurtosis()), 6),
            "p10"     : round(float(s.quantile(0.10)), 6),
            "p25"     : round(float(s.quantile(0.25)), 6),
            "median"  : round(float(s.median()), 6),
            "p75"     : round(float(s.quantile(0.75)), 6),
            "p90"     : round(float(s.quantile(0.90)), 6),
            "jb_p"    : round(float(jb_p), 8),
        }

    chatgpt_results[label] = {
        "cutoff_year" : CHATGPT_CUTOFF,
        "pre"         : grp_stats(pre),
        "post"        : grp_stats(post),
        "ks_stat"     : round(float(ks_stat), 6) if ks_stat is not None else None,
        "ks_p"        : round(float(ks_p), 8)    if ks_p    is not None else None,
    }

with open(OUTPUTS / "chatgpt_comparison.json", "w") as f:
    json.dump(chatgpt_results, f, indent=2)
print("Saved outputs/chatgpt_comparison.json")

print("\nDone. All outputs saved to outputs/")
