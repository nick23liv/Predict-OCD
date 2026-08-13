"""
visualize_cognition.py
======================
Produces two figures exploring longitudinal raw cognitive trajectories:

  Figure 1 (trajectories.png)
    – Individual spaghetti lines (score vs age) for each of the 5 domains.
    – Population LMM trend + 95 % CI overlaid.
    – Points coloured by wave so the cross-sectional age shift is visible
      alongside the within-person trajectory.

  Figure 2 (same_vs_individual_slopes.png)
    – Model comparison: does everyone change at the same rate (Model A),
      or do participants differ in their rate of change (Model B)?
    – Three columns per domain: Model A trajectories (parallel), Model B
      trajectories (fanning), and a statistics box (ΔBIC, LRT, R², SD).
    – ΔBIC > 50 = very strong evidence for individual rates of change.

Usage
-----
# Use real data (default paths):
  python visualize_cognition.py

# Custom paths:
  python visualize_cognition.py \
      --input   /path/to/harmonised_cognition_ABCDv7.csv \
      --dyn     /dataset/abcd/v7/phenotype/ab_g_dyn.tsv \
      --out-dir /path/to/output/figures

Raw variables plotted (inputs to the z-scoring step):
  IQ                   → IQ_scaled_notz          (WISC-V scaled score, M=10 SD=3)
  Response Inhibition  → RespInhib_computed_notz  (NIH Toolbox Flanker; nihtb only)
  Cognitive Flexibility→ CogFlex_computed_notz    (NIH Toolbox DCCS computed score)
  Working Memory       → WorkingMem_raw_notz      (List Sorting raw score)
  Attention            → Attention_raw_notz        (Little Man Task accuracy)
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # headless-safe (HPC / no display)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf  # <-- Added for LMM support

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualise longitudinal cognitive trajectories.")
parser.add_argument(
    "--input",
    default="harmonised_cognition_ABCDv7.csv",
    help="Path to the harmonised cognition CSV.",
)
parser.add_argument(
    "--dyn",
    default=None,
    help="Path to ab_g_dyn.tsv (for visit age). "
         "Defaults to the same directory as --input, then to "
         "/dataset/abcd/v7/phenotype/ab_g_dyn.tsv.",
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="Directory for output PNGs. Defaults to the same directory as --input.",
)
args = parser.parse_args()

# Resolve paths
INPUT_FILE = args.input
INPUT_DIR  = os.path.dirname(os.path.abspath(INPUT_FILE))

if args.dyn:
    DYN_FILE = args.dyn
else:
    # Check local directory first, then TRE phenotype path
    local_dyn = os.path.join(INPUT_DIR, "ab_g_dyn.tsv")
    DYN_FILE  = local_dyn if os.path.exists(local_dyn) \
                else "/dataset/abcd/v7/phenotype/ab_g_dyn.tsv"

OUT_DIR = args.out_dir if args.out_dir else INPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Aesthetics ─────────────────────────────────────────────────────────────────
DOMAIN_COLORS = {
    "IQ":              "#4e79a7",   # blue
    "RespInhib":       "#f28e2b",   # orange
    "CogFlex":         "#59a14f",   # green
    "WorkingMem":      "#e15759",   # red
    "Attention":       "#76b7b2",   # teal
}

WAVE_COLORS = {
    "ses-00A": "#bde0f5",   # light powder blue
    "ses-01A": "#7fc4e8",   # sky blue
    "ses-02A": "#3da8d8",   # medium blue
    "ses-03A": "#1788be",   # azure
    "ses-04A": "#0d65a0",   # cobalt
    "ses-05A": "#094d80",   # dark blue
    "ses-06A": "#05365c",   # very dark blue
    "ses-07A": "#021d35",   # near-black navy
}
WAVE_LABELS = {
    "ses-00A": "Baseline (~9 y)",
    "ses-01A": "Wave 1 (~10 y)",
    "ses-02A": "Wave 2 (~11 y)",
    "ses-03A": "Wave 3 (~12 y)",
    "ses-04A": "Wave 4 (~13 y)",
    "ses-05A": "Wave 5 (~14 y)",
    "ses-06A": "Wave 6 (~15 y)",
    "ses-07A": "Wave 7 (~16 y)",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":     11,
    "axes.titlesize":    12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
})

# ── Domain configuration ───────────────────────────────────────────────────────
DOMAINS = {
    "IQ": {
        "col":    "IQ_scaled_notz",
        "title":  "IQ\n(WISC-V Scaled Score)",
        "ylabel": "Scaled score",
        "note":   "M=10, SD=3 normative scale",
        "filter": None,
    },
    "RespInhib": {
        "col":    "RespInhib_computed_notz",
        "title":  "Response Inhibition\n(NIH Toolbox Flanker)",
        "ylabel": "Computed score",
        "note":   "NIH Toolbox only (nihtb_flanker)",
        "filter": ("task_type", "nihtb_flanker"),
    },
    "CogFlex": {
        "col":    "CogFlex_computed_notz",
        "title":  "Cognitive Flexibility\n(NIH Toolbox DCCS)",
        "ylabel": "Computed score",
        "note":   "",
        "filter": None,
    },
    "WorkingMem": {
        "col":    "WorkingMem_raw_notz",
        "title":  "Working Memory\n(List Sorting Raw)",
        "ylabel": "Raw score",
        "note":   "",
        "filter": None,
    },
    "Attention": {
        "col":    "Attention_raw_notz",
        "title":  "Attention\n(Little Man Task Accuracy)",
        "ylabel": "Proportion correct",
        "note":   "Range 0–1",
        "filter": None,
    },
}

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
df  = pd.read_csv(INPUT_FILE, na_values="n/a", low_memory=False)
dyn = pd.read_csv(DYN_FILE,   sep="\t", na_values="n/a", low_memory=False)

age_df = (
    dyn[["participant_id", "session_id", "ab_g_dyn__visit_age"]]
    .dropna(subset=["ab_g_dyn__visit_age"])
    .drop_duplicates(subset=["participant_id", "session_id"], keep="first")
    .rename(columns={"ab_g_dyn__visit_age": "age"})
)
df = df.merge(age_df, on=["participant_id", "session_id"], how="left")

print(f"  {df['participant_id'].nunique()} participants, "
      f"{df['session_id'].nunique()} waves, "
      f"{df.shape[0]} total rows")


# ── Helper: OLS fit with 95 % CI band ─────────────────────────────────────────
def ols_with_ci(x, y, n_points=200, ci=0.95):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), n_points)
    y_pred  = slope * x_range + intercept
    n       = len(x)
    t_crit  = stats.t.ppf((1 + ci) / 2, df=n - 2)
    x_mean  = x.mean()
    SS_x    = np.sum((x - x_mean) ** 2)
    resid   = y - (slope * x + intercept)
    s       = np.sqrt(np.sum(resid ** 2) / (n - 2))
    se_fit  = s * np.sqrt(1 / n + (x_range - x_mean) ** 2 / SS_x)
    return x_range, y_pred, y_pred - t_crit * se_fit, y_pred + t_crit * se_fit, r, p, slope


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Individual trajectories
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 1: Individual trajectories …")

fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle(
    "Longitudinal Cognitive Trajectories (raw scores)\n"
    "Grey lines = individual participants; coloured line = population LMM fixed trend ± 95 % CI; "  # <-- Updated Title text
    "dots coloured by wave",
    fontsize=12, y=1.01,
)
axes1_flat = axes1.flatten()
axes1_flat[5].set_visible(False)

for ax_idx, (domain_key, domain) in enumerate(DOMAINS.items()):
    ax    = axes1_flat[ax_idx]
    col   = domain["col"]
    color = DOMAIN_COLORS[domain_key]

    data = df.copy()
    if domain["filter"]:
        fcol, fval = domain["filter"]
        data = data[data[fcol] == fval]

    plot_df = data[["participant_id", "session_id", "age", col]].dropna(subset=[col, "age"])

    # Draw spaghetti lines
    for pid, pdata in plot_df.groupby("participant_id"):
        pdata = pdata.sort_values("age")
        if len(pdata) >= 2:
            ax.plot(
                pdata["age"], pdata[col],
                color="#cccccc", alpha=0.05, linewidth=0.5, zorder=1,
            )

    # Draw wave scatters
    for wave in ["ses-00A", "ses-01A", "ses-02A", "ses-03A", "ses-04A", "ses-05A", "ses-06A", "ses-07A"]:
        wdata = plot_df[plot_df["session_id"] == wave]
        if not wdata.empty:
            ax.scatter(
                wdata["age"], wdata[col],
                color=WAVE_COLORS[wave], s=8, alpha=0.65,
                linewidths=0, zorder=2,
            )

    # ── MODIFIED: Fit LMM Population Line & Analytical 95% CI ──────────────────
    baseline_ages = plot_df.loc[plot_df["session_id"] == "ses-00A", "age"]
    baseline_mean_age = baseline_ages.mean() if not baseline_ages.empty else plot_df["age"].min()
    plot_df["age_centered"] = plot_df["age"] - baseline_mean_age

    # A random-intercept + random-slope model is unidentifiable when almost every
    # participant contributes a single observation (e.g. IQ, administered once).
    # In that case the "trend" is a purely CROSS-SECTIONAL age gradient between
    # different people, not within-person change — fit and label it as such
    # rather than passing it off as a longitudinal LMM trend.
    _n_subj_pre = plot_df["participant_id"].nunique()
    _obs_per_subj = len(plot_df) / _n_subj_pre if _n_subj_pre else 0
    is_cross_sectional = _obs_per_subj < 1.1

    if is_cross_sectional:
        fit = ols_with_ci(plot_df["age"].values, plot_df[col].values)
        if fit:
            x_range, y_pred, y_lo, y_hi, r, p, slope = fit
            ax.plot(x_range, y_pred, color=color, linewidth=2.5, zorder=4,
                    linestyle="--",
                    label=f"Cross-sectional trend (Δ={slope:+.2f}/yr)")
            ax.fill_between(x_range, y_lo, y_hi, alpha=0.25, color="#e0e0e0", zorder=3)
        ax.text(
            0.02, 0.06,
            "⚠ Single timepoint per participant —\n"
            "cross-sectional gradient, NOT within-person change",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#b35806",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4e6",
                      edgecolor="#f0b27a", alpha=0.9),
        )
        result = None
    else:
        try:
            model = smf.mixedlm(f"{col} ~ age_centered", plot_df,
                                groups="participant_id", re_formula="~age_centered")
            result = model.fit(maxiter=300, method="powell")

            fixed_intercept = result.params["Intercept"]
            fixed_slope = result.params["age_centered"]

            x_range_raw = np.linspace(plot_df["age"].min(), plot_df["age"].max(), 200)
            x_range_centered = x_range_raw - baseline_mean_age
            y_pred = fixed_intercept + fixed_slope * x_range_centered

            # SE of the fixed population line from the analytical covariance matrix
            X_mat = np.column_stack([np.ones_like(x_range_centered), x_range_centered])
            cov_fe = result.cov_params().loc[
                ["Intercept", "age_centered"], ["Intercept", "age_centered"]].values
            se_fit = np.sqrt(np.diag(np.dot(np.dot(X_mat, cov_fe), X_mat.T)))
            y_lo = y_pred - 1.96 * se_fit
            y_hi = y_pred + 1.96 * se_fit

            ax.plot(x_range_raw, y_pred, color=color, linewidth=2.5, zorder=4,
                    label=f"LMM Fixed Trend (Δ={fixed_slope:+.2f}/yr)")
            ax.fill_between(x_range_raw, y_lo, y_hi, alpha=0.25, color="#e0e0e0", zorder=3)

        except Exception as e:
            # Fallback to pooled OLS if the mixed model fails to converge.
            print(f"  [warn] LMM failed for {col} ({e}); falling back to OLS.", flush=True)
            fit = ols_with_ci(plot_df["age"].values, plot_df[col].values)
            if fit:
                x_range, y_pred, y_lo, y_hi, r, p, slope = fit
                ax.plot(x_range, y_pred, color=color, linewidth=2.5, zorder=4,
                        linestyle=":", label=f"OLS fallback (Δ={slope:+.2f}/yr)")
                ax.fill_between(x_range, y_lo, y_hi, alpha=0.25, color="#e0e0e0", zorder=3)

    n_obs  = plot_df[col].notna().sum()
    n_subj = plot_df["participant_id"].nunique()
    ax.set_xlabel("Age (years)", labelpad=4)
    ax.set_ylabel(domain["ylabel"], labelpad=4)
    ax.set_title(domain["title"], fontweight="bold", pad=8)
    ax.set_xlim(plot_df["age"].min() - 0.3, plot_df["age"].max() + 0.3)
    ax.text(
        0.02, 0.97,
        f"n={n_subj} participants\n{n_obs} observations",
        transform=ax.transAxes, fontsize=8, va="top", color="#555555",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    if domain["note"]:
        ax.set_title(
            f"{domain['title']}\n({domain['note']})",
            fontweight="bold", pad=8, fontsize=10,
        )

legend_ax = axes1_flat[5]
legend_ax.set_visible(True)
legend_ax.axis("off")
handles = [
    mlines.Line2D([], [], color="#aaaaaa", linewidth=1.5, alpha=0.6,
                  label="Individual trajectory"),
    mpatches.Patch(color="#e0e0e0", alpha=0.8, label="Population 95 % CI"),
]
for wave in ["ses-00A", "ses-01A", "ses-02A", "ses-03A", "ses-04A", "ses-05A", "ses-06A", "ses-07A"]:
    handles.append(
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=WAVE_COLORS[wave],
                      markersize=8, label=WAVE_LABELS[wave])
    )
legend_ax.legend(
    handles=handles, loc="center", frameon=False,
    title="Legend", title_fontsize=10, fontsize=9,
)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "trajectories.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"  Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Do participants differ in their RATE of change, or only in where
#            they start?   Trajectories + model comparison, in one figure.
# ══════════════════════════════════════════════════════════════════════════════
#
# Two models are fitted per domain on the long-format data (one row per visit):
#
#   Model A : score ~ time + (1    | participant)
#             everyone changes at the SAME rate; participants differ only in
#             where they start  →  every trajectory has the same gradient, so
#             the lines are PARALLEL.
#   Model B : score ~ time + (time | participant)
#             participants also differ in their own rate of change  →  the
#             lines FAN OUT and cross.
#
# Statistics comparing the two models:
#   ΔBIC    difference in BIC (>10 strong, >50 very strong evidence for B)
#   LRT     likelihood ratio test, boundary-corrected (50:50 mix of χ²(1),
#           χ²(2), since testing a variance against zero sits on the edge of
#           the parameter space)
#   R²      share of the variance in the REAL observed scores each model
#           reproduces, pooled over all visits
#
# Models are compared by ML (not REML) because they differ in random-effects
# structure. Time is centred on each participant's own first visit.
# Minimum 1000 participants with ≥2 waves required — domains below this
# threshold are skipped with a console note.
print("\nBuilding Figure 2: same rate for everyone, or individual rates? …")

N_LINES = 150
_rs = np.random.default_rng(1)
# IQ has too few repeated measures in ABCD to fit meaningful random slopes.
SLOPE_DOMAINS = [d for d in DOMAINS if d != "IQ"]

fits2 = {}
for dom in SLOPE_DOMAINS:
    col = DOMAINS[dom]["col"]
    domain_filter = DOMAINS[dom].get("filter")

    d = df.copy()
    if domain_filter:
        fcol, fval = domain_filter
        d = d[d[fcol] == fval]

    d = d.dropna(subset=[col, "age"]).copy()
    cnt = d.groupby("participant_id").size()
    d = d[d["participant_id"].isin(cnt[cnt >= 2].index)]

    if d["participant_id"].nunique() < 1000:
        print(f"    [skip] {dom}: fewer than 1000 participants with ≥2 waves.")
        continue

    print(f"    -> {dom} …", flush=True)
    d["t0"]   = d.groupby("participant_id")["age"].transform("min")
    d["time"] = d["age"] - d["t0"]

    try:
        mA = smf.mixedlm(f"{col} ~ time", d, groups="participant_id",
                         re_formula="~1").fit(reml=False, maxiter=300, method="powell")
        mB = smf.mixedlm(f"{col} ~ time", d, groups="participant_id",
                         re_formula="~time").fit(reml=False, maxiter=300, method="powell")
    except Exception as e:
        print(f"    [skip] {dom}: model fit failed ({e}).")
        continue

    lrt  = 2 * (mB.llf - mA.llf)
    pval = 0.5 * stats.chi2.sf(lrt, 1) + 0.5 * stats.chi2.sf(lrt, 2)
    n    = len(d)
    dbic = (-2 * mA.llf + 4 * np.log(n)) - (-2 * mB.llf + 6 * np.log(n))

    reA, reB = mA.random_effects, mB.random_effects
    pA = np.array([mA.params["Intercept"] + reA[p].iloc[0] + mA.params["time"] * t
                   for p, t in zip(d["participant_id"], d["time"])])
    pB = np.array([mB.params["Intercept"] + reB[p].iloc[0]
                   + (mB.params["time"] + reB[p].iloc[1]) * t
                   for p, t in zip(d["participant_id"], d["time"])])
    y   = d[col].values
    sst = ((y - y.mean()) ** 2).sum()

    fits2[dom] = {
        "d": d, "mA": mA, "mB": mB, "lrt": lrt, "p": pval, "dbic": dbic,
        "r2A": 1 - ((y - pA) ** 2).sum() / sst,
        "r2B": 1 - ((y - pB) ** 2).sum() / sst,
        "sd_slope": np.sqrt(max(mB.cov_re.iloc[1, 1], 0)),
        "n_subj": d["participant_id"].nunique(),
    }

if not fits2:
    print("  [skip] No domains had sufficient data for model comparison.")
else:
    doms2 = list(fits2.keys())
    fig2, axes2 = plt.subplots(len(doms2), 3, figsize=(16.2, 4.7 * len(doms2)),
                               squeeze=False,
                               gridspec_kw={"width_ratios": [1, 1, 0.62]})
    fig2.suptitle(
        "Do participants differ in their RATE of change, or only in where they start?\n"
        "Each line = one participant's model-implied trajectory, coloured by starting level",
        fontsize=12, y=1.005,
    )

    for row, dom in enumerate(doms2):
        f = fits2[dom]
        d, mA, mB = f["d"], f["mA"], f["mB"]
        pids = list(mB.random_effects.keys())
        pick = _rs.choice(pids, size=min(N_LINES, len(pids)), replace=False)
        tt   = np.linspace(0, d["time"].quantile(0.98), 50)
        starts = np.array([mB.params["Intercept"] + mB.random_effects[p].iloc[0]
                           for p in pick])
        span = starts.max() - starts.min()
        norm = (starts - starts.min()) / span if span > 0 else np.full_like(starts, 0.5)
        cmap = plt.get_cmap("coolwarm")

        for k, mdl in enumerate(["A", "B"]):
            ax = axes2[row, k]
            for p, nv in zip(pick, norm):
                if mdl == "A":
                    b0 = mA.params["Intercept"] + mA.random_effects[p].iloc[0]
                    b1 = mA.params["time"]
                else:
                    b0 = mB.params["Intercept"] + mB.random_effects[p].iloc[0]
                    b1 = mB.params["time"] + mB.random_effects[p].iloc[1]
                ax.plot(tt, b0 + b1 * tt, color=cmap(nv), alpha=0.45, linewidth=0.9)
            # Population average uses THIS panel's own model fixed effects.
            m_pop = mA if mdl == "A" else mB
            ax.plot(tt, m_pop.params["Intercept"] + m_pop.params["time"] * tt,
                    color="#111111", linewidth=2.6, label="population average")
            ax.set_xlabel("Years since that participant's first visit")
            ax.set_ylabel(DOMAINS[dom]["ylabel"], fontsize=9)
            ttl = ("Model A — ONE shared rate of change\n(lines parallel: nobody overtakes)"
                   if mdl == "A" else
                   "Model B — INDIVIDUAL rates of change\n(lines fan out and cross)")
            ax.set_title(f"{dom}\n{ttl}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7.5, frameon=False, loc="best")
            ax.spines[["top", "right"]].set_visible(False)

        ylo = min(axes2[row, 0].get_ylim()[0], axes2[row, 1].get_ylim()[0])
        yhi = max(axes2[row, 0].get_ylim()[1], axes2[row, 1].get_ylim()[1])
        axes2[row, 0].set_ylim(ylo, yhi)
        axes2[row, 1].set_ylim(ylo, yhi)

        # ΔBIC is signed: positive favours Model B (individual rates),
        # NEGATIVE favours Model A (one shared rate). The ladder is symmetric
        # so a strongly negative value is reported as evidence FOR Model A,
        # not mislabelled as a tie.
        _db = f["dbic"]
        verdict = ("MODEL B IS BETTER"             if _db >  50 else
                   "Model B better"                if _db >  10 else
                   "little to choose between them" if _db > -10 else
                   "Model A better"                if _db > -50 else
                   "MODEL A IS BETTER")
        # Green = strong evidence for individual rates; orange = strong evidence
        # against them; grey = inconclusive.
        _edge = ("#2ca02c" if _db > 50 else
                 "#d95f02" if _db < -10 else
                 "#cccccc")
        ax_txt = axes2[row, 2]
        ax_txt.axis("off")
        # At large n the LRT can be significant while BIC still favours the
        # simpler model — the effect is real but too small to justify the extra
        # parameters. Flag that disagreement explicitly rather than leaving the
        # reader to reconcile a significant p against a negative ΔBIC.
        _note = ""
        if _db < 0 and f["p"] < 0.05:
            _note = ("\nNOTE: LRT significant but ΔBIC negative —\n"
                     "effect is real but too small to be worth\n"
                     "the extra parameters at this n.")

        ax_txt.text(
            0.0, 0.5,
            f"{verdict}\n"
            f"ΔBIC = {_db:+.0f}   (positive favours B, negative favours A;\n"
            f"          |ΔBIC|>10 strong, >50 very strong)\n"
            f"LRT χ² = {f['lrt']:.0f},  p = {f['p']:.0e}\n"
            f"variance of real scores explained:\n"
            f"    {f['r2A']*100:.1f}%  →  {f['r2B']*100:.1f}%   "
            f"({(f['r2B']-f['r2A'])*100:+.1f} pts)\n"
            f"SD of individual rates = {f['sd_slope']:.4f}/yr\n"
            f"n = {f['n_subj']} participants"
            f"{_note}",
            transform=ax_txt.transAxes, fontsize=8.8, va="center", ha="left",
            fontweight="normal", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=_edge, linewidth=1.8, alpha=0.93),
        )

    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "same_vs_individual_slopes.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {out2}")

    print("\n" + "=" * 74)
    print("SAME RATE FOR EVERYONE (A)  vs  INDIVIDUAL RATES (B)")
    print("=" * 74)
    print(f"{'Domain':<13}{'subj':>6}{'dBIC':>8}{'LRT':>8}{'p':>10}"
          f"{'R2 A':>8}{'R2 B':>8}{'gain':>8}   {'verdict'}")
    print("-" * 74)
    for dom in doms2:
        f = fits2[dom]
        _db = f["dbic"]
        v = ("B (very strong)" if _db >  50 else
             "B"               if _db >  10 else
             "inconclusive"    if _db > -10 else
             "A"               if _db > -50 else
             "A (very strong)")
        print(f"{dom:<13}{f['n_subj']:>6}{_db:>+8.0f}{f['lrt']:>8.0f}"
              f"{f['p']:>10.0e}{f['r2A']:>8.3f}{f['r2B']:>8.3f}"
              f"{(f['r2B']-f['r2A'])*100:>+7.1f}pp   {v}")
    print("\nΔBIC is signed: POSITIVE favours Model B (individual rates of change),")
    print("NEGATIVE favours Model A (one shared rate). |ΔBIC| > 50 = very strong.")
    print("A significant LRT with a negative ΔBIC means the effect is real but too")
    print("small to justify the extra parameters at this sample size.")
    print("=" * 74)

print("\nDone.")
