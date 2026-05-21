"""
Streamlit app — Wage as Signal: Labor Demand Results
Run locally:  streamlit run code/app.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wage as Signal", layout="wide")

# ── Load results ───────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    with open(OUTPUTS / "regression_results.json") as f:
        return json.load(f)

@st.cache_data
def load_panel(slug):
    return pd.read_csv(OUTPUTS / f"panel_{slug}.csv")

results = load_results()
spec_labels = list(results.keys())


# ── Helpers ────────────────────────────────────────────────────────────────
def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt_coef(v, p):
    s = stars(p)
    return f"{v:.4f}{s}" if abs(v) > 0.0001 else f"{v:.2e}{s}"


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.title("Wage as Signal")
st.sidebar.markdown("Labor demand estimation")
st.sidebar.markdown("---")

selected = st.sidebar.selectbox("Specification", spec_labels, index=1)
r = results[selected]

slug_map = {
    "(1) Levels — All wages"       : "1_levels",
    "(2) Log wage — All wages"     : "2_log_wage",
    "(3) Levels — Observed wages"  : "3_levels",
    "(4) Log wage — Observed wages": "4_log_wage",
}
panel = load_panel(slug_map[selected])
w_col = r["w_col"]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:**\n"
    r"$$l_i = \alpha + \hat{\beta}\,w_{-i} + \hat{\epsilon}_i$$"
)
st.sidebar.markdown(
    "- $l_i$: log posting count  \n"
    "- $w_{-i}$: leave-one-out mean wage  \n"
    "- $\\hat{\\epsilon}_i = \\tau a_i$: residual  \n"
)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
st.title("Labor Demand Estimation")
st.markdown(
    f"**Spec:** {selected} &nbsp;|&nbsp; "
    f"**N** = {r['n']:,} firm-occupation-year cells &nbsp;|&nbsp; "
    f"**R²** = {r['r2']:.4f}"
)

# ── Row 1: regression coefficients ────────────────────────────────────────
st.markdown("### Regression coefficients")
col1, col2, col3 = st.columns(3)

col1.metric(
    "Constant α",
    fmt_coef(r["const_coef"], r["const_p"]),
    f"SE = {r['const_se']:.4f}",
)
col2.metric(
    "β on w₋ᵢ",
    fmt_coef(r["w_coef"], r["w_p"]),
    f"SE = {r['w_se']:.2e}  p = {r['w_p']:.3f}",
)
col3.metric("R²", f"{r['r2']:.4f}")

st.markdown(
    "<small>\\* p<0.10 &nbsp; \\*\\* p<0.05 &nbsp; \\*\\*\\* p<0.01 &nbsp; "
    "(HC3 robust SEs)</small>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Row 2: scatter + residual distribution ─────────────────────────────────
st.markdown("### Residual distribution  $\\tau a_i$")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(panel[w_col], panel["log_count"],
               alpha=0.4, s=18, color="steelblue", label="Data")
    x_line = np.linspace(panel[w_col].min(), panel[w_col].max(), 200)
    y_line = r["const_coef"] + r["w_coef"] * x_line
    ax.plot(x_line, y_line, "r-", lw=1.8, label="OLS fit")
    ax.set_xlabel("$w_{-i}$" if w_col == "w_loo" else "$\\log w_{-i}$")
    ax.set_ylabel("$\\log$ posting count")
    ax.set_title("Data and fitted line")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close()

with right:
    resid = panel["residual"]
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Histogram + normal
    x_grid = np.linspace(resid.min(), resid.max(), 300)
    mu, sigma = stats.norm.fit(resid)
    axes[0].hist(resid, bins=30, density=True, color="steelblue", alpha=0.65)
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma),
                 "r--", lw=1.5, label="Normal")
    axes[0].set_xlabel(r"$\hat{\epsilon}_i$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histogram")
    axes[0].legend(fontsize=8)

    # Q-Q
    stats.probplot(resid, plot=axes[1])
    axes[1].set_title("Q-Q plot")
    axes[1].get_lines()[1].set(color="red", lw=1.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Row 3: residual stats table ────────────────────────────────────────────
st.markdown("---")
st.markdown("### Residual statistics")

resid_table = pd.DataFrame({
    "Statistic": ["Std dev", "Skewness", "Ex. kurtosis",
                  "p10", "p25", "Median", "p75", "p90",
                  "Jarque-Bera p-value"],
    "Value": [
        f"{r['resid_std']:.4f}",
        f"{r['resid_skew']:.4f}",
        f"{r['resid_kurt']:.4f}",
        f"{r['resid_p10']:.4f}",
        f"{r['resid_p25']:.4f}",
        f"{r['resid_med']:.4f}",
        f"{r['resid_p75']:.4f}",
        f"{r['resid_p90']:.4f}",
        f"{r['jb_p']:.2e}",
    ]
})
st.dataframe(resid_table, use_container_width=False, hide_index=True, width=320)

# ── Row 4: all-specs comparison ────────────────────────────────────────────
st.markdown("---")
with st.expander("Compare all specifications"):
    comp = []
    for lbl, res in results.items():
        comp.append({
            "Spec"            : lbl,
            "β coef"          : fmt_coef(res["w_coef"], res["w_p"]),
            "β p-value"       : f"{res['w_p']:.3f}",
            "R²"              : f"{res['r2']:.4f}",
            "N"               : f"{res['n']:,}",
            "Resid skewness"  : f"{res['resid_skew']:.3f}",
            "JB p-value"      : f"{res['jb_p']:.2e}",
        })
    st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)
