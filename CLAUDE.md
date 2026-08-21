# CLAUDE.md — Wages as Signals of AI Complementarity

**Repo purpose:** Revelio Labs microdata → firm × occupation × month panel testing the co-authors'
signal-extraction labor-demand model, and characterizing how the residual (a firm's private
AI-complementarity signal) shifts after ChatGPT.

**Status (2026-08-21):** Pilot proven on sample data (`code/app.py`, `outputs/`). Full pipeline —
Revelio raw → panel → estimation — not yet built; the pipeline's code will live in this repo, but the
Revelio raw data itself does not (§5). This file is the single source of truth for scope, definitions,
and conventions — **update it whenever a decision is made**, don't let decisions live only in commits
or chat.

**Read this whole file before writing pipeline code.**

---

## 1. The research question

Firms hold noisy private beliefs about whether AI complements or substitutes for labor in an
occupation. Market wages, formed in equilibrium, are a *public* signal of that belief. The model's
striking result: **labor demand is increasing in the wage** — a high wage is a cost, but also news
that AI makes the labor valuable, and the second effect dominates in the co-authors' calibration.
Estimating the demand equation gives a positive wage coefficient; its **residual is the firm's private
signal**, and the distribution of that residual — mean, variance, cross-occupation covariance —
changes after a large common shock to beliefs (ChatGPT, 2022-11-30).

Two contributions, same object at two levels of aggregation — build the panel once, both fall out:

| | Contribution | Panel needs |
|---|---|---|
| C1 | Wage-as-signal: positive wage coefficient; residual distribution shifts/widens post-ChatGPT | firm×occ×month: log postings, LOO occupation wage, pre/post indicator |
| C2 | Occupation covariance: AI hits occupations in correlated bundles | N×N covariance of the *same* residuals, cross-occupation |

## 2. Theory → estimating equation (scalar model — the baseline)

```
l_i = α + β·w_{-i} + ε_i,      ε_i = τ·a_i
```
- `w_{-i}` = leave-one-out log market wage in the firm's occupation-month (own-firm postings
  excluded — required by theory, not just an endogeneity fix: firms are measure-zero in the model,
  and in the richer iteration `w` and the residual share a common aggregate shock, so LOO is the only
  fix).
- `w` is **contemporaneous** — the model is static/simultaneous, don't lag it. Robustness: `t-1`,
  3-month trailing mean.
- **`β > 0` is the prediction, not a data-quality failure.** Pilot: `β̂ = 0.241`, SE 0.115, p = 0.037,
  R² = 0.0064, N = 691 firm-occupation-**year** cells, HC3 SEs (`outputs/regression_results.json`,
  spec "(2) Log wage — All wages").
- Residual moments are structural: `E[ε] = τ·a` (economy-wide complementarity), `Var(ε) = τ`
  (precision of *private* signals — not the public/wage-signal precision; that intuition runs the
  opposite direction and is wrong here). **Prediction: `Var(ε)` rises post-ChatGPT, `E[ε]` shifts.**

**Matrix version** (N occupations, C2): residual becomes `(1/ρ_i)Σ_i⁻¹μ_i`, an N-vector per firm; its
cross-occupation covariance is C2's object. `ρ_i` = firm risk aversion, enters multiplicatively — see
§9 threat list.

## 3. Confirmed decisions

| Decision | Value |
|---|---|
| Occupation unit | Revelio `onet_code` (8-digit, predicted). `role_k150` fallback if coverage/quality poor. **Pilot uses `role_k50`, coarser than either — that's a pilot simplification, not the target spec.** |
| Wage sample | **Posted salary only** (`salary_predicted = FALSE`). No all-wages variant carried forward. ~25% of postings qualify. Position-file `salary` is never a wage observation (fully modeled, not posted). |
| Panel grain | firm × occupation × **month**. (Pilot is firm×occ×**year** — a pilot simplification, not the target grain.) |
| Treatment date | `post_t = 1{t ≥ 2022-12}` |
| `w` timing | contemporaneous, baseline |
| Residual construction | **fit β on pre-period only, project residuals out-of-sample (S3, §10)** — pooled-fit residuals are mechanically zero-sum across the split and understate the story; report pooled alongside as a secondary check, never as the only cut |
| Firm FE | **not used** in the scalar baseline — it absorbs `ε_i = τa_i` entirely by construction. Legitimate again only under the matrix/C2 estimation, and only after re-deriving the residual there in writing |
| Outcome | `log(postings)` primary; `net = hires − separations` as the model-faithful secondary (only outcome defined on zeros/negatives) |
| Firm identifier | `rcid`; `ultimate_parent_rcid` as a config toggle, ships in the postings file |

## 4. Data

**Revelio postings** (grain: one row/posting, verify in Stage 0): `job_id`, `rcid`,
`ultimate_parent_rcid`, `company`, `post_date`, `remove_date`, `title_raw`, `role_k1500`/`role_k150`,
`onet_code`, `salary`/`salary_predicted`/`salary_min`/`salary_max`, `country`/`state`/`metro_area`,
`seniority`, `remote_type`, `rics_k50`/`k200`/`k400`.

**Revelio positions** (grain: one row/person-position spell, ~1,000 files, map-reduce): `user_id`,
`rcid`, `startdate`/`enddate`, `title`/`role_kn`/`onet_code`, `seniority`, `education`. Drives
realized hires/separations (`net`) and worker-quality controls (§9).

**External:** BLS OES (wage benchmark), O\*NET (DWA task similarity for C2 validation), SOC
crosswalks, Eisfeldt-Schubert-Zhang-Taska GenAI exposure scores (**validation only — not a
regressor**; the model recovers `a` from residuals, it doesn't proxy it), BLS JOLTS×CES (public
replication, already built in `code/fetch_bls.py` / `run_bls_regression.py`).

## 5. Sample restrictions

- Country = US only; resolve the null-country rule in Stage 0.
- Dedup on `job_id`; collapse identical title+location reposts from the same firm within 7 days.
- Drop staffing agencies / job-board aggregators (NAICS 5613-equivalent + manual list) — their
  residual isn't their private belief, it's someone else's posting. [@Pablo, what do think about this?]
- **Postings only, keep `salary_predicted = FALSE`** — the ~4× cut that defines the sample (§3).
- Firm sample default: `rcid` posts in ≥12 months of the window **and** appears in both pre and post
  periods; report the unrestricted sample too.
- **Active occupation set** (zeros): balance the panel over each firm's occupations with ≥1 posting or
  employee in the pre-period — not all ~1,000 O\*NET codes. `log(postings)` drops every true zero cell,
  and the zero rate likely moved post-ChatGPT, so **report the zero rate by period** as a standing
  diagnostic; `net` (defined at zero/negative) is the check that selection isn't driving results.

## 6. Repo conventions

**Layout — extends what's already here, doesn't replace it:**
```
wage-as-signal/
├── CLAUDE.md              # this file
├── code/                  # existing: 00_explore, 01_labor_demand, app.py, fetch_bls.py, run_*.py
│   └── pipeline/          # NEW, proposed: s00_discover .. s08_output, one dir per pipeline stage
├── config/                # NEW: config.yaml (paths, window, thresholds), field_map.yaml
├── data/
│   ├── raw/               # existing, read-only — small external refs only (OES, O*NET, Eisfeldt), NOT Revelio
│   └── interim/           # NEW: pipeline stage outputs, parquet, gitignored
├── outputs/               # existing: tables, figures, results json
├── paper/                 # existing: Overleaf sync
└── tests/                 # NEW: pytest — panel + residual invariants (§11)
```
`code/pipeline/` and the stage layout are a **proposal — confirm before the first pipeline
PR**, since it extends the repo's current flat-`code/` convention.

**Where raw data lives:** Revelio postings + positions sit on an external cluster, ~14 TB across two
parquet directories — **not in this repo, and never committed here.** Point `config/config.yaml` at
the cluster mount. This repo holds pipeline code, small reference/external data, and outputs only.

**Stack:** Python 3.11+, Polars/DuckDB lazy-streaming for anything touching raw (never
`pl.read_parquet` on the raw directories — project to `field_map.yaml` columns only; at 14 TB that
alone is the single biggest win available). OLS+HC3 baseline, `pyfixest` for FE variants, `sklearn`
(Ledoit-Wolf/graphical lasso) for C2.

**Data safety (non-negotiable):** raw cluster dirs are read-only, always; never overwrite an existing
output path — error and ask; atomic writes (temp file + rename); log row counts at every
filter/join/aggregation; assert expected post-join row counts, halt on mismatch.

**The 14 TB consolidation pass (pipeline Stage 1) is long-running — these are requirements, not
suggestions:** per-file manifest with resume (`status, rows_in, rows_out, sha256, error`), atomic
per-file output, fail loudly on a malformed file but isolate it (job continues, stage exits non-zero
if anything failed), row reconciliation asserted at the end (raw = filtered-out + retained, no
unexplained loss), progress+ETA every 5% of files, `--limit-files N` dry run exercised before any
multi-day run.

**Governing trade-off:** efficiency matters, correctness matters more. Never sample to save time
without asking; never skip a validation because the input is big.

## 7. Writing pipeline code: correctness, sharding, and gates

**Correctness beats speed, always.** 

**Shard by construction, not as an afterthought.** Every stage that touches the cluster is designed
around independent, resumable units of work from the start, not retrofitted later:
- Pick the natural partition for the stage — one raw file (Stage 1), one `rcid` hash bucket
  (positions map-reduce, Stage 3), one occupation×month (Stage 4) — and write the stage so a single
  shard can run, fail, and be retried in isolation, with no shared mutable state across shards.
- Each shard writes its own output atomically (temp file → rename, §6) and is independently
  idempotent: re-running a finished shard reproduces the same output; re-running a failed shard never
  requires touching the ones that already succeeded.
- A stage's "combine" step (cross-file dedup, global filters, final write) is a separate, explicit
  pass over completed shards — never folded into per-shard logic. Anything that needs a global view
  is exactly what silently breaks if it's done shard-by-shard instead.
- This is the same manifest pattern already specified for Stage 1 (§6) — apply it to every
  cluster-touching stage, not just the consolidation pass.

**Two gates, both required, before any code runs against real data or the cluster:**

1. **Tests, written from this file, not from the implementation.** For every stage: output schema,
   row-count/shape expectations, no-duplicate checks on key columns, join fan-out (output rows ≤ left
   table on a left join), no nulls in non-nullable columns, boundary cases (empty input, single-row
   input, all-null column). Tests are written against §3/§5/§10's definitions and invariants — someone
   working from this file alone, without seeing the implementation, should be able to write them.
2. **Adversarial review, context bounded to this file plus the code.** The reviewer just this CLAUDE.md — nothing else: no chat history, no task description, no other session
   context. Brief: could this step silently drop, duplicate, or corrupt rows; are join keys validated
   and row counts checked post-join; are dtypes assumed without validation; could this overwrite or
   clobber existing output; does it fail gracefully on malformed/empty input; is anything here
   inconsistent with a decision in §3, a restriction in §5, or an invariant in §10. The review is a
   compliance check against this file, not a generic code read — that's what the bounded context is
   for. Every issue raised gets resolved, documented inline, before the code runs — especially before
   it's pointed at the cluster.

Neither gate substitutes for the other: tests catch what the spec says should be true and isn't;
review catches what the spec didn't think to test.

## 8. Pipeline stages

| Stage | Does | Output | Acceptance |
|---|---|---|---|
| 0 Discovery | profile schema, resolve grain/dedup key, coverage by source/month, `onet_code` vintage, `salary_predicted` incidence by state/industry/size, panel size, runtime estimate | `data_dictionary.md`, `field_map.yaml` | sign-off before Stage 1 |
| 1 Consolidation | one full pass: project → filter (US, sample window, `salary_predicted=FALSE`) → normalize → write | `interim/postings/`, `interim/positions/` | manifest all-done, row accounting reconciles exactly |
| 2 Occupation | normalize/crosswalk `onet_code`, coverage report, merge Eisfeldt for validation only | `occ_crosswalk.parquet` | ≥90% posting-weighted coverage |
| 3 Flows | map-reduce positions → firm×occ×month hires/seps, truncate backfill lag | `firm_occ_month_flows.parquet` | headcount corr ≥0.9 with posting volume |
| 4 Market wage | `w_{j,t}` and LOO `w_{-i,j,t}`, composition-adjusted variant | `market_wage.parquet` | corr ≥0.7 with BLS OES |
| 5 Panel | balance over each firm's active occupation set, attach outcomes/controls/`post_t` | `panel.parquet` + codebook | invariants (§11) pass; power calc run before estimation |
| 6 Estimation | S1–S6 (§10) | tables | `β̂>0` reproduced at scale |
| 7 Covariance | C2, occupation set locked *here* — not before | `ai_complementarity_covariance_v1.csv` | known occupation bundles appear |
| 8 Output | tables/figures/dataset release | — | reproduces via one command |

**C2 occupation selection is deferred to Stage 7**, after the route (cross-firm vs. cross-time
covariance, or both) is decided — each imposes a different feasibility bar (co-occurrence count vs.
monthly density), so picking occupations first risks optimizing for the wrong criterion.

## 9. Identification threats and tests

| Threat | Test |
|---|---|
| Dropped zeros differ pre/post (log-postings conditions on `postings>0`) | report zero rate by period; re-run on `net` |
| Reverse causality / simultaneity | LOO handles it in theory; check largest-firm posting share per occ×month cell, flag/exclude dominant-firm cells |
| Worker-quality shift masquerading as signal | composition-adjusted wage; direct quality controls from positions (seniority, degree, prior-employer rank) — **run before writing any claims** |
| Risk-aversion (`ρ_i`) dispersion confounding `Var(ε)` | estimate residual moments within industry (`rics_k50`) |
| **Posted-salary selection** (disclosure laws phase in near the treatment date — a composition shift, not measurement error, under posted-only) | **Colorado-only** (constant disclosure regime) as the clean test — run first; fixed-composition subsample; describe selection by state/industry/size; IPW reweight; bound under unobserved selection; this will be robustness for salary transparency issue |
| Vendor/source coverage break | measure coverage by month by source in Stage 0; set window to avoid a source entering/exiting mid-sample; single-source robustness |
| Pooled-fit zero-sum residuals | use S3 (pre-fit, projected), not pooled, as baseline |
| Pooled slope masking a slope change | S4: fit β separately pre/post, compare variance under each |

Falsification: placebo date 2021-12; placebo (low-exposure) occupations; 500-draw permutation of the
pre/post label.

## 10. Specifications

- **S1** `log_postings = α + β·W + ε` (HC3). Prediction `β>0`.
- **S2** Residual distribution pre/post, **pooled fit** — report but don't lead with it (see S3).
- **S3 — baseline.** Fit β pre-period only, project residuals to the post period. Fixes S2's zero-sum
  artifact (pooled pre/post means are mechanically linked: `n_pre·mean_pre + n_post·mean_post ≈ 0`).
  Firm-clustered bootstrap SE on the projection.
- **S4** Separate `β_pre`/`β_post`, compare `Var(ε)` under each slope — separates a `τ` (private
  precision, the estimand) change from a `τ_e` (public signal) change, which a single pooled slope
  conflates.
- **S5** FE ladder: industry×month FE safe; occupation×month FE **not estimable** (absorbs `W`); firm
  FE **not run** in this build (§3).
- **S6** Public replication (BLS JOLTS×CES, already in `code/`) — treat as illustration, not
  corroboration; two trending, procyclical series with industry FE will produce a large significant
  `β` on their own. Fixes to apply here: month FE, a stationarity test.

SEs: two-way cluster (firm, occ×month) for `β`; firm-clustered bootstrap for all variance/KS/moment
statistics — `scipy`'s Levene/Bartlett/F/KS all assume independence, and our cells cluster within firm.

## 11. Panel & residual invariants (assert in `tests/`)

`(rcid, onet_code, month)` unique · no month outside sample window · every non-null `W` has
≥`MIN_POSTINGS_FOR_WAGE` (default 20) other-firm postings · `post_t = 1{month≥2022-12}` exactly ·
every estimation-sample firm has both pre- and post-period obs · no regressor at `t` uses info dated
`>t` (contemporaneous `W` is fine — forbid the future, not the present) · outcome counts ≥0, `net` may
be negative · pooled-fit residuals: `Σ n_g·mean_g ≈ 0` across the split (a sanity check on *which* fit
produced a given residual set) · projected (S3) residuals: post-mean is free — assert it's *not*
pinned.

## 12. Known traps

- **Source coverage break.** Unknown which of LinkedIn/Indeed starts when in this delivery — measure
  in Stage 0 before setting the sample window.
- **Posted-salary selection timing.** Disclosure laws (CO Jan 2021, NYC Nov 2022, CA/WA Jan 2023) land
  right at the Nov 2022 treatment date. Colorado-only is the clean laboratory.
- **Pooled-fit zero-sum residuals** (§3, §10 S3) — the single easiest mistake to make and hardest to
  notice, since it doesn't error, it just turns ten industries' mean-shifts into one fact instead of
  ten.
- **Thin LOO cells.** Posted-only quarters the sample; `MIN_POSTINGS_FOR_WAGE` binds far more than on
  the full-wage universe. Report missingness by occ×month as a primary Stage 4 diagnostic.
- **`onet_code` is predicted, not observed** — classification error attenuates; eyeball the top 50 by
  volume for plausibility.
- **Position backfill lag** — recent months are undercounted; measure and truncate (typically 3-6 mo).
- **Variance tests assume independence** — bootstrap by firm, always (§10).

## 13. Open questions (blocking)

- Country field representation + null-country rule (Stage 0).
- `onet_code` vintage — O\*NET-SOC 2010 vs. 2019 (Eisfeldt is 2010-keyed). Blocks the *validation
  layer* only, not Stages 1–6.
- `code/pipeline/` layout (§6) — confirm before the first pipeline PR.

## 14. Glossary

`a` true complementarity (unobserved) · `a_i` firm i's private signal · `τ` private-signal precision,
**`Var(ε)=τ` is the estimand** · `τ_e` public (wage) signal precision · `ρ_i` firm risk aversion,
enters the residual as `1/ρ_i` · `ε_i=τ·a_i` **the residual — the paper's estimand** · LOO
leave-one-out · `rcid` Revelio firm id · S3 pre-period-fit-and-project (baseline residual
construction)
