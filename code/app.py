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

# ── Helper: renders the full pre/post comparison block ─────────────────────
def render_comparison(panel_data, cg_data, wage_label):
    pre_r  = panel_data[panel_data["year"] <= cg_data["cutoff_year"]]["residual"]
    post_r = panel_data[panel_data["year"] >  cg_data["cutoff_year"]]["residual"]
    ks_stat = cg_data.get("ks_stat")
    ks_p    = cg_data.get("ks_p")

    x_min  = min(pre_r.min(), post_r.min()) - 0.2
    x_max  = max(pre_r.max(), post_r.max()) + 0.2
    x_grid = np.linspace(x_min, x_max, 300)

    # ── Side-by-side panels ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(
        rf"Distribution of $\tau a_i$ — {wage_label}", fontsize=13
    )
    for ax, grp, color, title in zip(
        axes,
        [pre_r, post_r],
        ["steelblue", "darkorange"],
        [f"Pre-ChatGPT  (year ≤ 2022,  N = {len(pre_r)})",
         f"Post-ChatGPT  (year ≥ 2023,  N = {len(post_r)})"],
    ):
        mu_g, sig_g = stats.norm.fit(grp)
        kde_g = stats.gaussian_kde(grp)
        ax.hist(grp, bins=20, density=True, color=color, alpha=0.4, label="Histogram")
        ax.plot(x_grid, kde_g(x_grid),                       color=color, lw=2.5, label="KDE")
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu_g, sig_g), color="red", lw=1.5,
                ls="--", label="Normal fit")
        ax.axvline(grp.mean(),   color=color,   lw=1.5, ls="--", label=f"Mean = {grp.mean():.2f}")
        ax.axvline(grp.median(), color="black", lw=1.2, ls=":",  label=f"Median = {grp.median():.2f}")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"Residual  $\tau a_i$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.text(0.97, 0.97,
                f"Skewness = {grp.skew():.2f}\nStd dev  = {grp.std():.2f}\nEx. kurt = {grp.kurtosis():.2f}",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        f"Each panel shows the OLS residual τaᵢ estimated using {wage_label.lower()}. "
        f"Histogram, KDE (solid), and normal fit (red dashed) are shown. "
        f"Dashed lines = group means; dotted lines = medians. Stats annotated top-right."
    )

    # ── Overlay ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for grp, color, lbl in [
        (pre_r,  "steelblue",  f"Pre-ChatGPT  (≤ 2022,  N={len(pre_r)})"),
        (post_r, "darkorange", f"Post-ChatGPT  (≥ 2023,  N={len(post_r)})"),
    ]:
        kde_g = stats.gaussian_kde(grp)
        ax.plot(x_grid, kde_g(x_grid), color=color, lw=2.5, label=lbl)
        ax.fill_between(x_grid, kde_g(x_grid), alpha=0.15, color=color)
        ax.axvline(grp.mean(), color=color, lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel(r"Residual  $\tau a_i$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(rf"Overlay — {wage_label}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        "Same residuals overlaid on one axis. Where the orange curve sits to the right of "
        "the blue curve, post-ChatGPT firms post more jobs than predicted relative to pre-ChatGPT."
    )

    # ── Tests ──────────────────────────────────────────────────────────────
    lev_stat,  lev_p  = stats.levene(pre_r, post_r, center="mean")
    bart_stat, bart_p = stats.bartlett(pre_r, post_r)
    f_stat = pre_r.var() / post_r.var()
    f_p    = 2 * min(
        stats.f.cdf(f_stat, len(pre_r)-1, len(post_r)-1),
        stats.f.sf( f_stat, len(pre_r)-1, len(post_r)-1),
    )

    def sig(p): return "✓ Reject H₀" if p < 0.05 else "✗ Fail to reject"

    tests_df = pd.DataFrame({
        "Test": [
            "KS test (equal distributions)",
            "Levene's test (equal variances)",
            "Bartlett's test (equal variances)",
            "F-test (equal variances)",
        ],
        "H₀": [
            "Same distribution",
            "Var(pre) = Var(post)",
            "Var(pre) = Var(post)",
            "Var(pre) = Var(post)",
        ],
        "Statistic": [
            f"{ks_stat:.4f}",
            f"{lev_stat:.4f}",
            f"{bart_stat:.4f}",
            f"{f_stat:.4f}",
        ],
        "p-value": [
            f"{ks_p:.4f}",
            f"{lev_p:.4f}",
            f"{bart_p:.4f}",
            f"{f_p:.4f}",
        ],
        "Result (5%)": [sig(ks_p), sig(lev_p), sig(bart_p), sig(f_p)],
    })

    pre_stats_d  = cg_data["pre"]
    post_stats_d = cg_data["post"]
    summary_df = pd.DataFrame({
        "Statistic"    : ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
        "Pre-ChatGPT"  : [pre_stats_d.get("n","—"),  f"{pre_stats_d.get('mean',0):.3f}",
                          f"{pre_stats_d.get('std',0):.3f}",  f"{pre_stats_d.get('skew',0):.3f}",
                          f"{pre_stats_d.get('kurt',0):.3f}", f"{pre_stats_d.get('median',0):.3f}"],
        "Post-ChatGPT" : [post_stats_d.get("n","—"), f"{post_stats_d.get('mean',0):.3f}",
                          f"{post_stats_d.get('std',0):.3f}",  f"{post_stats_d.get('skew',0):.3f}",
                          f"{post_stats_d.get('kurt',0):.3f}", f"{post_stats_d.get('median',0):.3f}"],
    })

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Summary statistics**")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
    with col_r:
        st.markdown("**Statistical tests**")
        st.dataframe(tests_df, hide_index=True, use_container_width=True)
        st.caption(
            "Levene's test is robust to non-normality (preferred here). "
            "Bartlett's assumes normality. F-test is two-sided."
        )


# ── Section 1: Observed wages only ────────────────────────────────────────
st.markdown("---")
st.markdown("### $\\tau a_i$ pre/post ChatGPT — employer-posted wages only (non-imputed)")
st.markdown("Split at October 2022: pre = year ≤ 2022, post = year ≥ 2023.")
obs_panel_data = load_panel("4_log_wage")
cg_obs = chatgpt["(4) Log wage — Observed wages"]
render_comparison(obs_panel_data, cg_obs, "Employer-posted wages only")

# ── Section 2: All wages (including imputed) ───────────────────────────────
st.markdown("---")
st.markdown("### $\\tau a_i$ pre/post ChatGPT — all wages (including Revelio imputations)")
st.markdown("Same split. Residuals from the regression on all 691 firm–occupation–year cells.")
all_panel_data = load_panel("2_log_wage")
cg_all = chatgpt["(2) Log wage — All wages"]
render_comparison(all_panel_data, cg_all, "All wages (including imputed)")

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC DATA: BLS JOLTS × CES
# ══════════════════════════════════════════════════════════════════════════════

def _bls_section(res_path, panel_path, section_title, subtitle):
    """Render one BLS JOLTS×CES section (reused for 4-industry and 10-industry)."""
    st.markdown("---")
    st.markdown(f"## {section_title}")
    st.markdown(subtitle)

    if not res_path.exists() or not panel_path.exists():
        st.info("Results not yet generated.")
        return

    with open(res_path) as f:
        r = json.load(f)
    panel = pd.read_csv(panel_path, parse_dates=["date"])

    # ── top metrics ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    c1.metric("R² (interaction model)", f"{r['r2_overall']:.4f}")
    c2.metric(
        "N",
        f"{r['n']:,}",
        f"{r['n_industries']} industries × monthly  ({r['date_min']} – {r['date_max']})",
    )

    # ── per-industry coefficient table ────────────────────────────────────
    st.markdown("### Wage coefficients by industry")
    st.markdown(
        r"Model: $\log(\text{openings}_{it}) = \alpha_i + \beta_i\,\log(\text{wage}_{it}) + \varepsilon_{it}$"
        "  —  estimated jointly with industry FE × wage interactions (HC3 SEs)."
    )

    def _stars(p):
        return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

    coef_rows = []
    for ind, d in r["industries"].items():
        coef_rows.append({
            "Industry":       ind,
            "β (log wage)":   f"{d['w_coef']:+.4f}{_stars(d['w_p'])}",
            "SE":             f"{d['w_se']:.4f}",
            "p-value":        f"{d['w_p']:.3f}",
            "R²":             f"{d['r2']:.4f}",
            "N":              d["n"],
        })
    coef_df = pd.DataFrame(coef_rows)
    st.dataframe(coef_df, hide_index=True, use_container_width=True)
    st.markdown(
        "<small>\\* p<0.10 &nbsp; \\*\\* p<0.05 &nbsp; \\*\\*\\* p<0.01 &nbsp;"
        " (HC3 robust SEs)</small>",
        unsafe_allow_html=True,
    )

    # ── scatter with per-industry fit lines ───────────────────────────────
    st.markdown("---")
    st.markdown("### Full-sample fit")
    left_b, right_b = st.columns(2)

    with left_b:
        industries = sorted(panel["industry"].unique())
        cmap       = plt.cm.get_cmap("tab20", len(industries))
        ind_colors = {ind: cmap(i) for i, ind in enumerate(industries)}

        fig, ax = plt.subplots(figsize=(5, 4))
        for ind, grp in panel.groupby("industry"):
            color = ind_colors[ind]
            ax.scatter(grp["log_wage"], grp["log_openings"],
                       color=color, s=10, alpha=0.45, label=ind)
            d    = r["industries"][ind]
            xs   = np.linspace(grp["log_wage"].min(), grp["log_wage"].max(), 80)
            ax.plot(xs, d["const_coef"] + d["w_coef"] * xs, color=color, lw=1.4)
        ax.set_xlabel(r"$\log(\text{wage}_{it})$", fontsize=11)
        ax.set_ylabel(r"$\log(\text{openings}_{it})$", fontsize=11)
        ax.set_title("Data + industry-specific OLS lines", fontsize=11)
        ax.legend(fontsize=5.5, ncol=2, loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "Each dot = one industry–month. Solid lines = industry-specific OLS fits "
            f"(β_i from table above). N = {r['n']:,}."
        )

    with right_b:
        resid  = panel["residual"]
        x_grid = np.linspace(resid.min(), resid.max(), 300)
        mu, sigma = stats.norm.fit(resid)

        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        axes[0].hist(resid, bins=30, density=True, color="steelblue", alpha=0.65)
        axes[0].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "r--", lw=1.5,
                     label="Normal")
        axes[0].set_xlabel(r"$\varepsilon_{it}$")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Histogram (pooled residuals)")
        axes[0].legend(fontsize=8)
        stats.probplot(resid, plot=axes[1])
        axes[1].set_title("Q-Q plot")
        axes[1].get_lines()[1].set(color="red", lw=1.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"Pooled residuals across all industries. "
            f"Skewness = {r['resid_skew']:.2f},  JB p = {r['jb_p']:.2e}."
        )

    # ── pre/post ChatGPT ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### $\\varepsilon_{it}$ pre vs post ChatGPT  (split: November 2022)")
    st.markdown("Pre = industry–months before Nov 2022.  Post = Nov 2022 onward.")

    cutoff_ts = pd.Timestamp(r["cutoff"])
    pre_r     = panel.loc[panel["date"] <  cutoff_ts, "residual"]
    post_r    = panel.loc[panel["date"] >= cutoff_ts, "residual"]

    x_lo  = min(pre_r.min(), post_r.min()) - 0.2
    x_hi  = max(pre_r.max(), post_r.max()) + 0.2
    xg    = np.linspace(x_lo, x_hi, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(rf"Distribution of $\varepsilon_{{it}}$ — {section_title}", fontsize=13)
    for ax, grp, color, title in zip(
        axes,
        [pre_r, post_r],
        ["steelblue", "darkorange"],
        [f"Pre-ChatGPT  (before Nov 2022,  N = {len(pre_r)})",
         f"Post-ChatGPT  (Nov 2022 onward,  N = {len(post_r)})"],
    ):
        mu_g, sig_g = stats.norm.fit(grp)
        kde_g = stats.gaussian_kde(grp)
        ax.hist(grp, bins=25, density=True, color=color, alpha=0.4, label="Histogram")
        ax.plot(xg, kde_g(xg),                       color=color, lw=2.5, label="KDE")
        ax.plot(xg, stats.norm.pdf(xg, mu_g, sig_g), color="red",  lw=1.5,
                ls="--", label="Normal fit")
        ax.axvline(grp.mean(),   color=color,   lw=1.5, ls="--",
                   label=f"Mean = {grp.mean():.3f}")
        ax.axvline(grp.median(), color="black", lw=1.2, ls=":",
                   label=f"Median = {grp.median():.3f}")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(r"Residual $\varepsilon_{it}$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.text(0.97, 0.97,
                f"Skewness = {grp.skew():.2f}\nStd dev  = {grp.std():.2f}\n"
                f"Ex. kurt = {grp.kurtosis():.2f}",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    for grp, color, lbl in [
        (pre_r,  "steelblue",  f"Pre-ChatGPT  (N={len(pre_r)})"),
        (post_r, "darkorange", f"Post-ChatGPT  (N={len(post_r)})"),
    ]:
        kde_g = stats.gaussian_kde(grp)
        ax.plot(xg, kde_g(xg), color=color, lw=2.5, label=lbl)
        ax.fill_between(xg, kde_g(xg), alpha=0.15, color=color)
        ax.axvline(grp.mean(), color=color, lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel(r"Residual $\varepsilon_{it}$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(rf"Overlay — {section_title}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        "Where the orange curve sits to the right of the blue curve, "
        "post-ChatGPT industry–months show higher job openings than market wages predict."
    )

    # ── statistical tests ─────────────────────────────────────────────────
    lev_stat,  lev_p  = stats.levene(pre_r, post_r, center="mean")
    bart_stat, bart_p = stats.bartlett(pre_r, post_r)
    f_stat = pre_r.var() / post_r.var()
    f_p    = 2 * min(
        stats.f.cdf(f_stat, len(pre_r)-1, len(post_r)-1),
        stats.f.sf( f_stat, len(pre_r)-1, len(post_r)-1),
    )

    def _sig(p): return "✓ Reject H₀" if p < 0.05 else "✗ Fail to reject"

    tests_df = pd.DataFrame({
        "Test": ["KS test (equal distributions)",
                 "Levene's test (equal variances)",
                 "Bartlett's test (equal variances)",
                 "F-test (equal variances)"],
        "H₀":  ["Same distribution",
                 "Var(pre) = Var(post)",
                 "Var(pre) = Var(post)",
                 "Var(pre) = Var(post)"],
        "Statistic": [f"{r['ks_stat']:.4f}", f"{lev_stat:.4f}",
                      f"{bart_stat:.4f}",    f"{f_stat:.4f}"],
        "p-value":   [f"{r['ks_p']:.4f}",   f"{lev_p:.4f}",
                      f"{bart_p:.4f}",       f"{f_p:.4f}"],
        "Result (5%)": [_sig(r["ks_p"]), _sig(lev_p), _sig(bart_p), _sig(f_p)],
    })

    pre_s, post_s = r["pre"], r["post"]
    summary_df = pd.DataFrame({
        "Statistic":    ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
        "Pre-ChatGPT":  [pre_s["n"],  f"{pre_s['mean']:.3f}",  f"{pre_s['std']:.3f}",
                         f"{pre_s['skew']:.3f}",  f"{pre_s['kurt']:.3f}",
                         f"{pre_s['median']:.3f}"],
        "Post-ChatGPT": [post_s["n"], f"{post_s['mean']:.3f}", f"{post_s['std']:.3f}",
                         f"{post_s['skew']:.3f}", f"{post_s['kurt']:.3f}",
                         f"{post_s['median']:.3f}"],
    })

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Summary statistics**")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
    with col_r:
        st.markdown("**Statistical tests**")
        st.dataframe(tests_df, hide_index=True, use_container_width=True)
        st.caption(
            "Levene's test is robust to non-normality (preferred). "
            "Bartlett's assumes normality. F-test is two-sided."
        )


_bls_section(
    OUTPUTS / "bls_regression_results.json",
    OUTPUTS / "bls_panel_with_residuals.csv",
    "Public Data Replication: BLS JOLTS × CES (4 industries)",
    "Second-best version using fully public data, 4 JOLTS industries available on FRED. "
    r"Unit of analysis: **industry × month** (monthly 2012–2024). "
    "Job openings from JOLTS; wages (avg hourly earnings) from CES. "
    r"Each industry gets its own intercept $\alpha_i$ and wage slope $\beta_i$."
)


# ══════════════════════════════════════════════════════════════════════════════


# ==============================================================================
# SECTION 8 - PUBLIC DATA (ALL 10 INDUSTRIES): BLS JOLTS x CES
# ==============================================================================

_bls_section(
    OUTPUTS / "bls10_regression_results.json",
    OUTPUTS / "bls10_panel_with_residuals.csv",
    "Full Industry Coverage: BLS JOLTS x CES (10 industries)",
    "Same specification extended to all 10 private-sector JOLTS industries "
    "via BLS API v2. "
    r"N = 1,560 (10 industries x 156 months, Jan 2012-Dec 2024). "
    r"Each industry gets its own intercept alpha_i and wage slope beta_i."
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PER-INDUSTRY RESIDUAL DISTRIBUTIONS (pre vs post ChatGPT)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## Residual shift by industry: pre vs post ChatGPT")
st.markdown(
    "Each panel shows the KDE of $\\varepsilon_{it}$ before (blue) and after "
    "(orange) November 2022 for one industry, using the industry-specific "
    "residuals from the 10-industry model above. "
    "Dashed verticals = group means. "
    "**Right shift** (orange to the right of blue) means that industry posted "
    "more openings than its own wage trend predicted after ChatGPT."
)

_bls10_panel_path = OUTPUTS / "bls10_panel_with_residuals.csv"

if not _bls10_panel_path.exists():
    st.info("10-industry panel not found.")
else:
    @st.cache_data
    def load_bls10_panel_sec9():
        return pd.read_csv(_bls10_panel_path, parse_dates=["date"])

    p10 = load_bls10_panel_sec9()
    CUTOFF_TS = pd.Timestamp("2022-11-01")

    industries_sorted = sorted(p10["industry"].unique())
    n_ind = len(industries_sorted)
    ncols = 5
    nrows = (n_ind + ncols - 1) // ncols   # ceil division

    SHORT = {
        "Trade, Transportation, Utilities":   "Trade/Transport/Util",
        "Professional and Business Services": "Prof. & Business Svcs",
        "Education and Health Services":      "Education & Health",
        "Mining and Logging":                 "Mining & Logging",
        "Financial Activities":               "Financial Activities",
        "Leisure and Hospitality":            "Leisure & Hospitality",
        "Other Services":                     "Other Services",
        "Information":                        "Information",
        "Manufacturing":                      "Manufacturing",
        "Construction":                       "Construction",
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.4))
    axes = axes.flatten()

    for idx, ind in enumerate(industries_sorted):
        ax   = axes[idx]
        grp  = p10[p10["industry"] == ind]["residual"]
        pre  = p10[(p10["industry"] == ind) & (p10["date"] <  CUTOFF_TS)]["residual"]
        post = p10[(p10["industry"] == ind) & (p10["date"] >= CUTOFF_TS)]["residual"]

        x_lo = min(pre.min(), post.min()) - 0.15
        x_hi = max(pre.max(), post.max()) + 0.15
        xg   = np.linspace(x_lo, x_hi, 300)

        for ser, color, lbl in [
            (pre,  "steelblue",  f"Pre  (N={len(pre)})"),
            (post, "darkorange", f"Post (N={len(post)})"),
        ]:
            kde = stats.gaussian_kde(ser)
            ax.plot(xg, kde(xg), color=color, lw=2, label=lbl)
            ax.fill_between(xg, kde(xg), alpha=0.15, color=color)
            ax.axvline(ser.mean(), color=color, lw=1.4, ls="--")

        delta = post.mean() - pre.mean()
        sign  = "▶" if delta > 0 else "◀"
        color_ann = "darkorange" if delta > 0 else "steelblue"
        ax.set_title(SHORT.get(ind, ind), fontsize=9, fontweight="bold")
        ax.text(0.97, 0.97,
                f"{sign} Δmean = {delta:+.3f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color=color_ann,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.set_xlabel(r"$\varepsilon_{it}$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(x_lo, x_hi)
        if idx % ncols == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")

    # hide any unused axes
    for idx in range(n_ind, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        r"Industry-level $\varepsilon_{it}$ distributions — pre (blue) vs post (orange) ChatGPT",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # summary table: direction and magnitude of shift
    rows = []
    for ind in industries_sorted:
        pre  = p10[(p10["industry"] == ind) & (p10["date"] <  CUTOFF_TS)]["residual"]
        post = p10[(p10["industry"] == ind) & (p10["date"] >= CUTOFF_TS)]["residual"]
        delta = post.mean() - pre.mean()
        ks_s, ks_p = stats.ks_2samp(pre, post)
        rows.append({
            "Industry":          ind,
            "Pre mean":          f"{pre.mean():+.3f}",
            "Post mean":         f"{post.mean():+.3f}",
            "Δ mean (post−pre)": f"{delta:+.3f}",
            "Direction":         "→ right" if delta > 0 else "← left",
            "KS p-value":        f"{ks_p:.4f}",
            "KS sig.":           "***" if ks_p < .01 else "**" if ks_p < .05 else "*" if ks_p < .10 else "",
        })
    st.markdown("**Mean shift and KS test by industry**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Δ mean = post-ChatGPT mean residual minus pre-ChatGPT mean residual. "
        "Right shift → industry posted more than its wage trend predicted after Nov 2022. "
        "KS p-value tests equality of pre and post distributions."
    )
