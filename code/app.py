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
st.latex(r"l_i = \left(\frac{\tau_a}{1+\tau\tau_e}\right)\bar{a} + w_{-i} + \tau a_i")
st.markdown(
    "This is a simply beautiful equation. And labor demand increases with wages. "
    "That's not a typo, it is due to the fact that wages signal complementarity. "
    "It is a cost, but higher wages also signal that complementarity may be high. "
    "This equation can be estimated directly. $a_i$ can be estimated as noise."
)
st.markdown(
    f"**Spec:** {selected} &nbsp;|&nbsp; "
    f"**N** = {r['n']:,} firm-occupation-year cells &nbsp;|&nbsp; "
    f"**R²** = {r['r2']:.4f}"
)

# ── Section 1: Pre/post ChatGPT error distributions (observed wages only) ──
st.markdown("---")
st.markdown("### Distribution of errors $\\tau a_i$ — before and after ChatGPT")
st.markdown(
    "Residuals from the regression estimated on **employer-posted wages only** (non-imputed). "
    "Split at October 2022: pre = year ≤ 2022, post = year ≥ 2023."
)

# Always use the observed-wages / log-wage panel for this plot
obs_panel_data = load_panel("4_log_wage")
cg_obs  = chatgpt["(4) Log wage — Observed wages"]
pre_r   = obs_panel_data[obs_panel_data["year"] <= cg_obs["cutoff_year"]]["residual"]
post_r  = obs_panel_data[obs_panel_data["year"] >  cg_obs["cutoff_year"]]["residual"]
ks_stat = cg_obs.get("ks_stat")
ks_p    = cg_obs.get("ks_p")

x_min  = min(pre_r.min(), post_r.min()) - 0.2
x_max  = max(pre_r.max(), post_r.max()) + 0.2
x_grid = np.linspace(x_min, x_max, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
fig.suptitle(
    r"Distribution of $\tau a_i$ from regression on employer-posted wages",
    fontsize=13
)

for ax, grp, color, title in zip(
    axes,
    [pre_r,       post_r],
    ["steelblue", "darkorange"],
    [f"Pre-ChatGPT  (year ≤ 2022,  N = {len(pre_r)})",
     f"Post-ChatGPT  (year ≥ 2023,  N = {len(post_r)})"],
):
    mu_g, sig_g = stats.norm.fit(grp)
    kde_g = stats.gaussian_kde(grp)

    ax.hist(grp, bins=20, density=True, color=color, alpha=0.4, label="Histogram")
    ax.plot(x_grid, kde_g(x_grid),                      color=color, lw=2.5, label="KDE")
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu_g, sig_g), color="red", lw=1.5,
            ls="--", label="Normal fit")
    ax.axvline(grp.mean(),   color=color, lw=1.5, ls="--",
               label=f"Mean = {grp.mean():.2f}")
    ax.axvline(grp.median(), color="black", lw=1.2, ls=":",
               label=f"Median = {grp.median():.2f}")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"Residual  $\tau a_i$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=11, color=color, fontweight="bold")
    ax.legend(fontsize=8.5)

    # Annotate key stats inside the plot
    stats_txt = (
        f"Skewness = {grp.skew():.2f}\n"
        f"Std dev  = {grp.std():.2f}\n"
        f"Ex. kurt = {grp.kurtosis():.2f}"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
st.pyplot(fig)
plt.close()

ks_interp = (
    f"The Kolmogorov-Smirnov test **rejects** the null of equal distributions "
    f"(KS = {ks_stat:.3f}, p = {ks_p:.3f}), indicating the two distributions are "
    f"statistically different."
    if ks_p is not None and ks_p < 0.05 else
    f"The Kolmogorov-Smirnov test does **not** reject the null of equal distributions "
    f"(KS = {ks_stat:.3f}, p = {ks_p:.3f})."
    if ks_p is not None else ""
)
st.caption(
    f"Each panel shows the distribution of the OLS residual τaᵢ = $l_i - \\hat{{\\alpha}} - \\hat{{\\beta}}\\,w_{{-i}}$, "
    f"estimated using only the {len(obs_panel_data):,} firm–occupation–year cells where employers "
    f"actually posted a wage (non-imputed). "
    f"Histograms, kernel density estimates (solid), and fitted normal curves (red dashed) are shown. "
    f"Dashed vertical lines mark group means; dotted lines mark medians. "
    f"Key statistics are annotated in the top-right corner of each panel. "
    + ks_interp +
    f" A compression of the right tail post-2022 would be consistent with AI tools "
    f"reducing the signaling advantage of high-ability firms."
)

# Overlay plot — both curves on the same axes
kde_pre  = stats.gaussian_kde(pre_r)
kde_post = stats.gaussian_kde(post_r)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_grid, kde_pre(x_grid),  color="steelblue",  lw=2.5,
        label=f"Pre-ChatGPT  (≤ 2022,  N={len(pre_r)})")
ax.plot(x_grid, kde_post(x_grid), color="darkorange", lw=2.5,
        label=f"Post-ChatGPT  (≥ 2023,  N={len(post_r)})")
ax.fill_between(x_grid, kde_pre(x_grid),  alpha=0.15, color="steelblue")
ax.fill_between(x_grid, kde_post(x_grid), alpha=0.15, color="darkorange")
ax.axvline(pre_r.mean(),  color="steelblue",  lw=1.5, ls="--", alpha=0.8,
           label=f"Pre mean = {pre_r.mean():.2f}")
ax.axvline(post_r.mean(), color="darkorange", lw=1.5, ls="--", alpha=0.8,
           label=f"Post mean = {post_r.mean():.2f}")
ax.set_xlabel(r"Residual  $\tau a_i$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(r"Overlay: $\tau a_i$ before vs. after ChatGPT (employer-posted wages only)",
             fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.caption(
    "Same residuals as the panels above, now overlaid on a single axis for direct comparison. "
    "Where the orange curve (post-ChatGPT) sits to the left of the blue curve (pre-ChatGPT), "
    "firms post fewer jobs than predicted relative to before — and vice versa for the right tail."
)

# ── Section 2: ChatGPT era KS test summary ────────────────────────────────
st.markdown("---")
st.markdown("### ChatGPT era comparison")
st.markdown(
    "Split at **October 2022** (ChatGPT launched November 30, 2022). "
    "Panel is yearly, so pre = year ≤ 2022, post = year ≥ 2023."
)

pre_stats  = cg_obs["pre"]
post_stats = cg_obs["post"]

col_left, col_right = st.columns([1, 1])

with col_left:
    comp_df = pd.DataFrame({
        "Statistic"    : ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
        "Pre-ChatGPT"  : [
            pre_stats.get("n", "—"),
            f"{pre_stats.get('mean',   0):.3f}",
            f"{pre_stats.get('std',    0):.3f}",
            f"{pre_stats.get('skew',   0):.3f}",
            f"{pre_stats.get('kurt',   0):.3f}",
            f"{pre_stats.get('median', 0):.3f}",
        ],
        "Post-ChatGPT" : [
            post_stats.get("n", "—"),
            f"{post_stats.get('mean',   0):.3f}",
            f"{post_stats.get('std',    0):.3f}",
            f"{post_stats.get('skew',   0):.3f}",
            f"{post_stats.get('kurt',   0):.3f}",
            f"{post_stats.get('median', 0):.3f}",
        ],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

with col_right:
    ks_sig = "✓ Reject H₀" if ks_p < 0.05 else "✗ Fail to reject H₀"
    st.markdown("**Kolmogorov-Smirnov test** — H₀: same distribution")
    st.metric("KS statistic", f"{ks_stat:.4f}")
    st.metric("p-value",      f"{ks_p:.4f}", ks_sig)

# ── Section 3: Regression coefficients ────────────────────────────────────
st.markdown("---")
st.markdown("### Estimation results")
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

# ── Section 4: scatter + residual distribution ─────────────────────────────
st.markdown("---")
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
        f"The red line is the OLS fit (β̂ = {r['w_coef']:.2e}, p = {r['w_p']:.3f})."
    )

with right:
    resid  = panel["residual"]
    x_grid = np.linspace(resid.min(), resid.max(), 300)
    mu, sigma = stats.norm.fit(resid)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    axes[0].hist(resid, bins=30, density=True, color="steelblue", alpha=0.65)
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "r--", lw=1.5, label="Normal")
    axes[0].set_xlabel(r"$\hat{\epsilon}_i$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histogram")
    axes[0].legend(fontsize=8)

    stats.probplot(resid, plot=axes[1])
    axes[1].set_title("Q-Q plot")
    axes[1].get_lines()[1].set(color="red", lw=1.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        f"The residual τaᵢ captures unobserved firm heterogeneity not explained by the "
        f"occupation wage — interpreted in the model as the firm's signaling ability aᵢ. "
        f"The histogram overlays a fitted normal (red dashed); the Q-Q plot compares sample "
        f"quantiles against a normal reference line. "
        f"Skewness = {r['resid_skew']:.2f} and the Jarque-Bera test rejects normality "
        f"(p = {r['jb_p']:.2e}), suggesting a longer right tail — some firms post far more "
        f"jobs than occupation wages alone predict."
    )

# ── Section 5: residual stats table ───────────────────────────────────────
st.markdown("---")
st.markdown("### Residual statistics")
resid_table = pd.DataFrame({
    "Statistic": ["Std dev", "Skewness", "Ex. kurtosis",
                  "p10", "p25", "Median", "p75", "p90", "Jarque-Bera p-value"],
    "Value": [
        f"{r['resid_std']:.4f}", f"{r['resid_skew']:.4f}", f"{r['resid_kurt']:.4f}",
        f"{r['resid_p10']:.4f}", f"{r['resid_p25']:.4f}", f"{r['resid_med']:.4f}",
        f"{r['resid_p75']:.4f}", f"{r['resid_p90']:.4f}", f"{r['jb_p']:.2e}",
    ]
})
st.dataframe(resid_table, use_container_width=False, hide_index=True, width=320)

# ── Section 6: all-specs comparison ───────────────────────────────────────
st.markdown("---")
with st.expander("Compare all specifications"):
    comp = []
    for lbl, res in results.items():
        comp.append({
            "Spec"           : lbl,
            "β coef"         : fmt_coef(res["w_coef"], res["w_p"]),
            "β p-value"      : f"{res['w_p']:.3f}",
            "R²"             : f"{res['r2']:.4f}",
            "N"              : f"{res['n']:,}",
            "Resid skewness" : f"{res['resid_skew']:.3f}",
            "JB p-value"     : f"{res['jb_p']:.2e}",
        })
    st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)
