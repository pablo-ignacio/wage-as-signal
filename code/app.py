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

@st.cache_data
def load_chatgpt():
    with open(OUTPUTS / "chatgpt_comparison.json") as f:
        return json.load(f)

results  = load_results()
chatgpt  = load_chatgpt()
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
    w_label = "mean occupation wage" if w_col == "w_loo" else "log mean occupation wage"
    st.caption(
        f"Each point is one firm–occupation–year cell. "
        f"The horizontal axis shows the leave-one-out {w_label} — "
        f"the average wage posted by all other firms in the same occupation and year, "
        f"which removes the mechanical correlation between a firm's own wage and the market wage. "
        f"The red line is the OLS fit (β̂ = {r['w_coef']:.2e}, "
        f"p = {r['w_p']:.3f})."
    )

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
    st.caption(
        f"The residual τaᵢ captures unobserved firm heterogeneity not explained by the "
        f"occupation wage — interpreted in the model as the firm's signaling ability aᵢ. "
        f"The histogram (left) overlays a fitted normal distribution (red dashed). "
        f"The Q-Q plot (right) compares sample quantiles against a normal reference line: "
        f"systematic departures indicate non-normality. "
        f"Here skewness = {r['resid_skew']:.2f} and the Jarque-Bera test rejects normality "
        f"(p = {r['jb_p']:.2e}), suggesting the distribution has a longer right tail than "
        f"a normal — some firms post far more jobs than occupation wages alone predict."
    )

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

# ── Row 4: ChatGPT era comparison ─────────────────────────────────────────
st.markdown("---")
st.markdown("### ChatGPT era comparison")
st.markdown(
    "Split at **October 2022** (ChatGPT launched November 30, 2022). "
    "Panel is yearly, so pre = year ≤ 2022, post = year ≥ 2023."
)

cg = chatgpt[selected]
pre_r  = panel[panel["year"] <= cg["cutoff_year"]]["residual"]
post_r = panel[panel["year"] >  cg["cutoff_year"]]["residual"]

if len(pre_r) < 2 or len(post_r) < 2:
    st.warning("Not enough observations in one of the groups for this spec. Try a different specification.")
else:
    cg_left, cg_right = st.columns([2, 1])

    with cg_left:
        fig, ax = plt.subplots(figsize=(7, 4))

        x_min = min(pre_r.min(), post_r.min())
        x_max = max(pre_r.max(), post_r.max())
        x_grid = np.linspace(x_min, x_max, 300)

        # KDE curves
        kde_pre  = stats.gaussian_kde(pre_r)
        kde_post = stats.gaussian_kde(post_r)
        ax.plot(x_grid, kde_pre(x_grid),  color="steelblue", lw=2,
                label=f"Pre-ChatGPT (≤ 2022, N={len(pre_r)})")
        ax.plot(x_grid, kde_post(x_grid), color="darkorange", lw=2,
                label=f"Post-ChatGPT (≥ 2023, N={len(post_r)})")

        # Filled area under each curve
        ax.fill_between(x_grid, kde_pre(x_grid),  alpha=0.15, color="steelblue")
        ax.fill_between(x_grid, kde_post(x_grid), alpha=0.15, color="darkorange")

        # Vertical lines at means
        ax.axvline(pre_r.mean(),  color="steelblue",  lw=1.2, ls="--", alpha=0.8)
        ax.axvline(post_r.mean(), color="darkorange", lw=1.2, ls="--", alpha=0.8)

        ax.set_xlabel(r"Residual $\tau a_i$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(r"Distribution of $\tau a_i$ before and after ChatGPT", fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        ks_p_val = cg.get("ks_p")
        ks_interp = (
            f"The Kolmogorov-Smirnov test **rejects** the null of equal distributions "
            f"(p = {ks_p_val:.3f}), suggesting the emergence of ChatGPT is associated "
            f"with a statistically significant shift in unobserved firm heterogeneity."
            if ks_p_val is not None and ks_p_val < 0.05 else
            f"The Kolmogorov-Smirnov test **does not reject** the null of equal distributions "
            f"(p = {ks_p_val:.3f} > 0.05) for this specification."
            if ks_p_val is not None else ""
        )
        st.caption(
            f"Kernel density estimates of the residual τaᵢ split at October 2022, "
            f"just before ChatGPT launched (November 30, 2022). "
            f"Since the panel is at the yearly level, pre-ChatGPT covers postings from years ≤ 2022 "
            f"(N = {len(pre_r)}) and post-ChatGPT covers years ≥ 2023 (N = {len(post_r)}). "
            f"Dashed vertical lines mark the group means. "
            + ks_interp +
            f" A decrease in right-skewness after 2022 would be consistent with AI tools "
            f"compressing the signaling advantage of high-ability firms."
        )

    with cg_right:
        # KS test
        ks_stat = cg.get("ks_stat")
        ks_p    = cg.get("ks_p")
        st.markdown("**Kolmogorov-Smirnov test**")
        st.markdown("*H₀: same distribution*")
        if ks_stat is not None:
            ks_sig = "✓ Reject H₀" if ks_p < 0.05 else "✗ Fail to reject H₀"
            st.metric("KS statistic", f"{ks_stat:.4f}")
            st.metric("p-value", f"{ks_p:.4f}", ks_sig)

        st.markdown("---")

        # Side-by-side stats
        pre_stats  = cg["pre"]
        post_stats = cg["post"]
        comp_df = pd.DataFrame({
            "Statistic" : ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
            "Pre-ChatGPT"  : [
                pre_stats.get("n", "—"),
                f"{pre_stats.get('mean', 0):.3f}",
                f"{pre_stats.get('std',  0):.3f}",
                f"{pre_stats.get('skew', 0):.3f}",
                f"{pre_stats.get('kurt', 0):.3f}",
                f"{pre_stats.get('median', 0):.3f}",
            ],
            "Post-ChatGPT" : [
                post_stats.get("n", "—"),
                f"{post_stats.get('mean', 0):.3f}",
                f"{post_stats.get('std',  0):.3f}",
                f"{post_stats.get('skew', 0):.3f}",
                f"{post_stats.get('kurt', 0):.3f}",
                f"{post_stats.get('median', 0):.3f}",
            ],
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ── Row 5: all-specs comparison ────────────────────────────────────────────
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
