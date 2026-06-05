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

st.markdown("---")
st.markdown("## Public Data Replication: BLS JOLTS × CES")
st.markdown(
    "Second-best version using fully public data. "
    r"Unit of analysis: **industry × month** (≈ 11 sectors, monthly 2012–2024). "
    "Job openings from [JOLTS](https://www.bls.gov/jlt/); "
    "wages (avg hourly earnings) from [CES](https://www.bls.gov/ces/). "
    "Same equation estimated: $\\log(\\text{openings}_{it}) = \\alpha + \\beta\\,\\log(\\text{wage}_{it}) + \\varepsilon_{it}$."
)

_bls_res_path   = OUTPUTS / "bls_regression_results.json"
_bls_panel_path = OUTPUTS / "bls_panel_with_residuals.csv"

if not _bls_res_path.exists() or not _bls_panel_path.exists():
    st.info(
        "BLS results not yet generated. Run from the project root:\n\n"
        "```\npython code/fetch_bls.py\npython code/run_bls_regression.py\n```"
    )
else:
    @st.cache_data
    def load_bls_results():
        with open(_bls_res_path) as f:
            return json.load(f)

    @st.cache_data
    def load_bls_panel():
        return pd.read_csv(_bls_panel_path, parse_dates=["date"])

    bls_r     = load_bls_results()
    bls_panel = load_bls_panel()

    # ── regression summary ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "β on log(wage)",
        fmt_coef(bls_r["w_coef"], bls_r["w_p"]),
        f"SE = {bls_r['w_se']:.4f}  p = {bls_r['w_p']:.3f}",
    )
    c2.metric("R²", f"{bls_r['r2']:.4f}")
    c3.metric(
        "N",
        f"{bls_r['n']:,}",
        f"{bls_r['n_industries']} industries × monthly  "
        f"({bls_r['date_min']} – {bls_r['date_max']})",
    )
    st.markdown(
        "<small>\\* p<0.10 &nbsp; \\*\\* p<0.05 &nbsp; \\*\\*\\* p<0.01 &nbsp;"
        " (HC3 robust SEs)</small>",
        unsafe_allow_html=True,
    )

    # ── scatter + residual distribution (full sample) ─────────────────────
    st.markdown("---")
    st.markdown("### Full-sample fit")
    left_b, right_b = st.columns(2)

    with left_b:
        industries = sorted(bls_panel["industry"].unique())
        cmap = plt.cm.get_cmap("tab20", len(industries))
        ind_colors = {ind: cmap(i) for i, ind in enumerate(industries)}

        fig, ax = plt.subplots(figsize=(5, 4))
        for ind, grp in bls_panel.groupby("industry"):
            ax.scatter(grp["log_wage"], grp["log_openings"],
                       color=ind_colors[ind], s=10, alpha=0.5, label=ind)
        x_line = np.linspace(bls_panel["log_wage"].min(), bls_panel["log_wage"].max(), 200)
        y_line = bls_r["const_coef"] + bls_r["w_coef"] * x_line
        ax.plot(x_line, y_line, "k-", lw=1.8, label="OLS fit")
        ax.set_xlabel(r"$\log(\text{wage}_{it})$", fontsize=11)
        ax.set_ylabel(r"$\log(\text{openings}_{it})$", fontsize=11)
        ax.set_title("Data and fitted line (by industry)", fontsize=11)
        ax.legend(fontsize=6, ncol=2, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"Each dot is one industry–month cell. Colors = industries. "
            f"Black line: OLS fit (β = {bls_r['w_coef']:+.4f}, p = {bls_r['w_p']:.3f}). "
            f"N = {bls_r['n']:,}."
        )

    with right_b:
        resid  = bls_panel["residual"]
        x_grid = np.linspace(resid.min(), resid.max(), 300)
        mu, sigma = stats.norm.fit(resid)

        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        axes[0].hist(resid, bins=30, density=True, color="steelblue", alpha=0.65)
        axes[0].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "r--", lw=1.5,
                     label="Normal")
        axes[0].set_xlabel(r"$\varepsilon_{it}$")
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
            f"Residual τaᵢₜ = unexplained posting intensity relative to market wage. "
            f"Skewness = {bls_r['resid_skew']:.2f}, "
            f"JB p = {bls_r['jb_p']:.2e}."
        )

    # ── pre/post ChatGPT comparison ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### $\\varepsilon_{it}$ pre vs post ChatGPT  (split: November 2022)")
    st.markdown("Pre = industry–months before Nov 2022.  Post = Nov 2022 onward.")

    cutoff_ts = pd.Timestamp(bls_r["cutoff"])
    pre_r_b   = bls_panel.loc[bls_panel["date"] <  cutoff_ts, "residual"]
    post_r_b  = bls_panel.loc[bls_panel["date"] >= cutoff_ts, "residual"]

    x_min_b = min(pre_r_b.min(), post_r_b.min()) - 0.2
    x_max_b = max(pre_r_b.max(), post_r_b.max()) + 0.2
    x_grid_b = np.linspace(x_min_b, x_max_b, 300)

    # side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(r"Distribution of $\varepsilon_{it}$ — BLS JOLTS × CES", fontsize=13)
    for ax, grp, color, title in zip(
        axes,
        [pre_r_b, post_r_b],
        ["steelblue", "darkorange"],
        [f"Pre-ChatGPT  (before Nov 2022,  N = {len(pre_r_b)})",
         f"Post-ChatGPT  (Nov 2022 onward,  N = {len(post_r_b)})"],
    ):
        mu_g, sig_g = stats.norm.fit(grp)
        kde_g = stats.gaussian_kde(grp)
        ax.hist(grp, bins=20, density=True, color=color, alpha=0.4, label="Histogram")
        ax.plot(x_grid_b, kde_g(x_grid_b),                        color=color, lw=2.5,
                label="KDE")
        ax.plot(x_grid_b, stats.norm.pdf(x_grid_b, mu_g, sig_g),  color="red",  lw=1.5,
                ls="--", label="Normal fit")
        ax.axvline(grp.mean(),   color=color,   lw=1.5, ls="--",
                   label=f"Mean = {grp.mean():.3f}")
        ax.axvline(grp.median(), color="black", lw=1.2, ls=":",
                   label=f"Median = {grp.median():.3f}")
        ax.set_xlim(x_min_b, x_max_b)
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
    st.caption(
        "Histogram, KDE (solid), and normal fit (red dashed) for each sub-period. "
        "Dashed lines = means; dotted = medians."
    )

    # overlay
    fig, ax = plt.subplots(figsize=(8, 4))
    for grp, color, lbl in [
        (pre_r_b,  "steelblue",  f"Pre-ChatGPT  (N={len(pre_r_b)})"),
        (post_r_b, "darkorange", f"Post-ChatGPT  (N={len(post_r_b)})"),
    ]:
        kde_g = stats.gaussian_kde(grp)
        ax.plot(x_grid_b, kde_g(x_grid_b), color=color, lw=2.5, label=lbl)
        ax.fill_between(x_grid_b, kde_g(x_grid_b), alpha=0.15, color=color)
        ax.axvline(grp.mean(), color=color, lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel(r"Residual $\varepsilon_{it}$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(r"Overlay — BLS JOLTS × CES", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        "Where the orange curve sits to the right of the blue curve, "
        "post-ChatGPT industry–months show higher job openings than market wages predict."
    )

    # stats tables + tests
    lev_stat,  lev_p  = stats.levene(pre_r_b, post_r_b, center="mean")
    bart_stat, bart_p = stats.bartlett(pre_r_b, post_r_b)
    f_stat = pre_r_b.var() / post_r_b.var()
    f_p = 2 * min(
        stats.f.cdf(f_stat, len(pre_r_b)-1, len(post_r_b)-1),
        stats.f.sf( f_stat, len(pre_r_b)-1, len(post_r_b)-1),
    )

    def sig_b(p): return "✓ Reject H₀" if p < 0.05 else "✗ Fail to reject"

    tests_df_b = pd.DataFrame({
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
            f"{bls_r['ks_stat']:.4f}",
            f"{lev_stat:.4f}",
            f"{bart_stat:.4f}",
            f"{f_stat:.4f}",
        ],
        "p-value": [
            f"{bls_r['ks_p']:.4f}",
            f"{lev_p:.4f}",
            f"{bart_p:.4f}",
            f"{f_p:.4f}",
        ],
        "Result (5%)": [
            sig_b(bls_r["ks_p"]),
            sig_b(lev_p),
            sig_b(bart_p),
            sig_b(f_p),
        ],
    })

    pre_s  = bls_r["pre"]
    post_s = bls_r["post"]
    summary_df_b = pd.DataFrame({
        "Statistic":    ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
        "Pre-ChatGPT":  [pre_s["n"],  f"{pre_s['mean']:.3f}",  f"{pre_s['std']:.3f}",
                         f"{pre_s['skew']:.3f}",  f"{pre_s['kurt']:.3f}",
                         f"{pre_s['median']:.3f}"],
        "Post-ChatGPT": [post_s["n"], f"{post_s['mean']:.3f}", f"{post_s['std']:.3f}",
                         f"{post_s['skew']:.3f}", f"{post_s['kurt']:.3f}",
                         f"{post_s['median']:.3f}"],
    })

    col_l_b, col_r_b = st.columns([1, 1])
    with col_l_b:
        st.markdown("**Summary statistics**")
        st.dataframe(summary_df_b, hide_index=True, use_container_width=True)
    with col_r_b:
        st.markdown("**Statistical tests**")
        st.dataframe(tests_df_b, hide_index=True, use_container_width=True)
        st.caption(
            "Levene's test is robust to non-normality (preferred). "
            "Bartlett's assumes normality. F-test is two-sided. "
            "Note: N is small (industry × month cells), so power is limited."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PUBLIC DATA (ALL 10 INDUSTRIES): BLS JOLTS × CES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## Full Industry Coverage: BLS JOLTS × CES (10 industries)")
st.markdown(
    "Same equation as above, now using all 10 private-sector JOLTS industries "
    r"(Construction, Information, Financial Activities, and Education & Health added). "
    "JOLTS series IDs fetched via BLS API v2 using registered key. "
    r"N = 1,560 (10 industries × 156 months, Jan 2012–Dec 2024)."
)

_bls10_res_path   = OUTPUTS / "bls10_regression_results.json"
_bls10_panel_path = OUTPUTS / "bls10_panel_with_residuals.csv"

if not _bls10_res_path.exists() or not _bls10_panel_path.exists():
    st.info("10-industry BLS results not yet generated.")
else:
    @st.cache_data
    def load_bls10_results():
        with open(_bls10_res_path) as f:
            return json.load(f)

    @st.cache_data
    def load_bls10_panel():
        return pd.read_csv(_bls10_panel_path, parse_dates=["date"])

    bls10_r     = load_bls10_results()
    bls10_panel = load_bls10_panel()

    # ── regression summary ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "β on log(wage)",
        fmt_coef(bls10_r["w_coef"], bls10_r["w_p"]),
        f"SE = {bls10_r['w_se']:.4f}  p = {bls10_r['w_p']:.3f}",
    )
    c2.metric("R²", f"{bls10_r['r2']:.4f}")
    c3.metric(
        "N",
        f"{bls10_r['n']:,}",
        f"{bls10_r['n_industries']} industries × monthly  "
        f"({bls10_r['date_min']} – {bls10_r['date_max']})",
    )
    st.markdown(
        "<small>\\* p<0.10 &nbsp; \\*\\* p<0.05 &nbsp; \\*\\*\\* p<0.01 &nbsp;"
        " (HC3 robust SEs)</small>",
        unsafe_allow_html=True,
    )

    # ── scatter + residual distribution ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Full-sample fit")
    left_c, right_c = st.columns(2)

    with left_c:
        industries10 = sorted(bls10_panel["industry"].unique())
        cmap10 = plt.cm.get_cmap("tab20", len(industries10))
        ind_colors10 = {ind: cmap10(i) for i, ind in enumerate(industries10)}

        fig, ax = plt.subplots(figsize=(5, 4))
        for ind, grp in bls10_panel.groupby("industry"):
            ax.scatter(grp["log_wage"], grp["log_openings"],
                       color=ind_colors10[ind], s=8, alpha=0.45, label=ind)
        x_line10 = np.linspace(bls10_panel["log_wage"].min(), bls10_panel["log_wage"].max(), 200)
        y_line10 = bls10_r["const_coef"] + bls10_r["w_coef"] * x_line10
        ax.plot(x_line10, y_line10, "k-", lw=1.8, label="OLS fit")
        ax.set_xlabel(r"$\log(\text{wage}_{it})$", fontsize=11)
        ax.set_ylabel(r"$\log(\text{openings}_{it})$", fontsize=11)
        ax.set_title("Data and fitted line (by industry)", fontsize=11)
        ax.legend(fontsize=5.5, ncol=2, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"Each dot is one industry–month cell. Colors = industries. "
            f"Black line: OLS fit (β = {bls10_r['w_coef']:+.4f}, p = {bls10_r['w_p']:.3f}). "
            f"N = {bls10_r['n']:,}."
        )

    with right_c:
        resid10 = bls10_panel["residual"]
        x_grid10 = np.linspace(resid10.min(), resid10.max(), 300)
        mu10, sigma10 = stats.norm.fit(resid10)

        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        axes[0].hist(resid10, bins=30, density=True, color="steelblue", alpha=0.65)
        axes[0].plot(x_grid10, stats.norm.pdf(x_grid10, mu10, sigma10), "r--", lw=1.5,
                     label="Normal")
        axes[0].set_xlabel(r"$\varepsilon_{it}$")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Histogram")
        axes[0].legend(fontsize=8)

        stats.probplot(resid10, plot=axes[1])
        axes[1].set_title("Q-Q plot")
        axes[1].get_lines()[1].set(color="red", lw=1.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"Residual τaᵢₜ = unexplained posting intensity relative to market wage. "
            f"Skewness = {bls10_r['resid_skew']:.2f}, "
            f"JB p = {bls10_r['jb_p']:.2e}."
        )

    # ── pre/post ChatGPT comparison ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### $\\varepsilon_{it}$ pre vs post ChatGPT  (split: November 2022)")
    st.markdown("Pre = industry–months before Nov 2022.  Post = Nov 2022 onward.")

    cutoff10 = pd.Timestamp(bls10_r["cutoff"])
    pre_r10  = bls10_panel.loc[bls10_panel["date"] <  cutoff10, "residual"]
    post_r10 = bls10_panel.loc[bls10_panel["date"] >= cutoff10, "residual"]

    x_min10  = min(pre_r10.min(), post_r10.min()) - 0.2
    x_max10  = max(pre_r10.max(), post_r10.max()) + 0.2
    x_grid10b = np.linspace(x_min10, x_max10, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(r"Distribution of $\varepsilon_{it}$ — BLS JOLTS × CES (10 industries)", fontsize=13)
    for ax, grp, color, title in zip(
        axes,
        [pre_r10, post_r10],
        ["steelblue", "darkorange"],
        [f"Pre-ChatGPT  (before Nov 2022,  N = {len(pre_r10)})",
         f"Post-ChatGPT  (Nov 2022 onward,  N = {len(post_r10)})"],
    ):
        mu_g, sig_g = stats.norm.fit(grp)
        kde_g = stats.gaussian_kde(grp)
        ax.hist(grp, bins=25, density=True, color=color, alpha=0.4, label="Histogram")
        ax.plot(x_grid10b, kde_g(x_grid10b),                        color=color, lw=2.5,
                label="KDE")
        ax.plot(x_grid10b, stats.norm.pdf(x_grid10b, mu_g, sig_g),  color="red",  lw=1.5,
                ls="--", label="Normal fit")
        ax.axvline(grp.mean(),   color=color,   lw=1.5, ls="--",
                   label=f"Mean = {grp.mean():.3f}")
        ax.axvline(grp.median(), color="black", lw=1.2, ls=":",
                   label=f"Median = {grp.median():.3f}")
        ax.set_xlim(x_min10, x_max10)
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
    st.caption(
        "Histogram, KDE (solid), and normal fit (red dashed) for each sub-period. "
        "Dashed lines = means; dotted = medians."
    )

    # overlay
    fig, ax = plt.subplots(figsize=(8, 4))
    for grp, color, lbl in [
        (pre_r10,  "steelblue",  f"Pre-ChatGPT  (N={len(pre_r10)})"),
        (post_r10, "darkorange", f"Post-ChatGPT  (N={len(post_r10)})"),
    ]:
        kde_g = stats.gaussian_kde(grp)
        ax.plot(x_grid10b, kde_g(x_grid10b), color=color, lw=2.5, label=lbl)
        ax.fill_between(x_grid10b, kde_g(x_grid10b), alpha=0.15, color=color)
        ax.axvline(grp.mean(), color=color, lw=1.5, ls="--", alpha=0.8)
    ax.set_xlabel(r"Residual $\varepsilon_{it}$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(r"Overlay — BLS JOLTS × CES (10 industries)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(
        "Where the orange curve sits to the right of the blue curve, "
        "post-ChatGPT industry–months show higher job openings than market wages predict."
    )

    # stats tables
    lev10_stat,  lev10_p  = stats.levene(pre_r10, post_r10, center="mean")
    bart10_stat, bart10_p = stats.bartlett(pre_r10, post_r10)
    f10_stat = pre_r10.var() / post_r10.var()
    f10_p = 2 * min(
        stats.f.cdf(f10_stat, len(pre_r10)-1, len(post_r10)-1),
        stats.f.sf( f10_stat, len(pre_r10)-1, len(post_r10)-1),
    )

    def sig10(p): return "✓ Reject H₀" if p < 0.05 else "✗ Fail to reject"

    tests_df10 = pd.DataFrame({
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
            f"{bls10_r['ks_stat']:.4f}",
            f"{lev10_stat:.4f}",
            f"{bart10_stat:.4f}",
            f"{f10_stat:.4f}",
        ],
        "p-value": [
            f"{bls10_r['ks_p']:.4f}",
            f"{lev10_p:.4f}",
            f"{bart10_p:.4f}",
            f"{f10_p:.4f}",
        ],
        "Result (5%)": [
            sig10(bls10_r["ks_p"]),
            sig10(lev10_p),
            sig10(bart10_p),
            sig10(f10_p),
        ],
    })

    pre10_s  = bls10_r["pre"]
    post10_s = bls10_r["post"]
    summary_df10 = pd.DataFrame({
        "Statistic":    ["N", "Mean", "Std dev", "Skewness", "Ex. kurtosis", "Median"],
        "Pre-ChatGPT":  [pre10_s["n"],  f"{pre10_s['mean']:.3f}",  f"{pre10_s['std']:.3f}",
                         f"{pre10_s['skew']:.3f}",  f"{pre10_s['kurt']:.3f}",
                         f"{pre10_s['median']:.3f}"],
        "Post-ChatGPT": [post10_s["n"], f"{post10_s['mean']:.3f}", f"{post10_s['std']:.3f}",
                         f"{post10_s['skew']:.3f}", f"{post10_s['kurt']:.3f}",
                         f"{post10_s['median']:.3f}"],
    })

    col_l10, col_r10 = st.columns([1, 1])
    with col_l10:
        st.markdown("**Summary statistics**")
        st.dataframe(summary_df10, hide_index=True, use_container_width=True)
    with col_r10:
        st.markdown("**Statistical tests**")
        st.dataframe(tests_df10, hide_index=True, use_container_width=True)
        st.caption(
            "Levene's test is robust to non-normality (preferred). "
            "Bartlett's assumes normality. F-test is two-sided."
        )
