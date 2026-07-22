"""
visualize_cognition.py
======================
Produces two figures exploring longitudinal raw cognitive trajectories:

  Figure 1 (trajectories.png)
    – Individual spaghetti lines (score vs age) for each of the 5 domains.
    – Population LMM trend + 95 % CI overlaid.
    – Points coloured by wave so the cross-sectional age shift is visible
      alongside the within-person trajectory.

  Figure 2 (baseline_vs_slope.png)
    – Scatter: each participant's baseline (ses-00A) score vs their
      individual rate of change (LMM slope, points/year).
    – Pearson r + p-value annotated.
    – Helps judge whether baseline alone predicts trajectory or whether
      longitudinal slope adds independent predictive power for ML.

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

import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # <-- Typo fixed here
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


# ── Helper: per-participant baseline + slope using LMM ──────────────────────────
def compute_baseline_slope(data, col):
    """
    Returns a DataFrame with one row per participant using a Linear Mixed-Effects Model (LMM):
      baseline  – model-estimated score at baseline age (fixed + random intercept)
      slope     – model-estimated rate of change (fixed + random slope)
      n_waves   – number of non-null timepoints used
    """
    model_data = data.dropna(subset=[col, "age", "participant_id"]).copy()
    if model_data.empty:
        return pd.DataFrame(columns=["participant_id", "baseline", "slope", "n_waves"])

    # Require ≥3 waves per participant for stable random slope estimation.
    # Participants with 1–2 waves are excluded from Figure 2 (not Figure 1).
    counts = model_data.groupby("participant_id").size()
    eligible = counts[counts >= 3].index
    model_data = model_data[model_data["participant_id"].isin(eligible)]

    if model_data["participant_id"].nunique() < 1000:
        print(f"    -> Skipping {col}: fewer than 1000 participants with ≥3 waves.")
        return pd.DataFrame(columns=["participant_id", "baseline", "slope", "n_waves"])

    counts = model_data.groupby("participant_id").size()
    n_subj = model_data["participant_id"].nunique()

    # Center age around baseline wave mean so that the Intercept represents baseline performance (~9yo)
    baseline_ages = model_data.loc[model_data["session_id"] == "ses-00A", "age"]
    baseline_mean_age = baseline_ages.mean() if not baseline_ages.empty else model_data["age"].min()
    model_data["age_centered"] = model_data["age"] - baseline_mean_age

    print(f"    -> Running LMM for {col} ({n_subj} subjects, {len(model_data)} rows)...", flush=True)

    try:
        model = smf.mixedlm(f"{col} ~ age_centered", model_data, groups="participant_id", re_formula="~age_centered")
        result = model.fit(maxiter=50, method="lbfgs")

        fixed_intercept = result.params["Intercept"]
        fixed_slope = result.params["age_centered"]
        re_effects = result.random_effects

        records = [
            {
                "participant_id": pid,
                "baseline": fixed_intercept + re_effects[pid]["participant_id"] if pid in re_effects else np.nan,
                "slope": fixed_slope + re_effects[pid]["age_centered"] if pid in re_effects else np.nan,
                "n_waves": counts[pid],
            }
            for pid in counts.index
        ]
            
    except Exception as e:
        print(f"  [Warning] LMM fit failed for {col} ({e}). Falling back to OLS loop.", flush=True)
        records = []
        for pid, grp in model_data.groupby("participant_id"):
            grp = grp.sort_values("age")
            baseline = grp.loc[grp["session_id"] == "ses-00A", col]
            baseline = baseline.iloc[0] if len(baseline) else np.nan
            slope = np.nan
            if len(grp) >= 2:
                res = stats.linregress(grp["age"].values, grp[col].values)
                slope = res.slope
            records.append({
                "participant_id": pid,
                "baseline": baseline,
                "slope": slope,
                "n_waves": len(grp),
            })

    return pd.DataFrame(records)


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

    try:
        model = smf.mixedlm(f"{col} ~ age_centered", plot_df, groups="participant_id", re_formula="~age_centered")
        result = model.fit(maxiter=50, method="lbfgs")

        fixed_intercept = result.params["Intercept"]
        fixed_slope = result.params["age_centered"]

        x_range_raw = np.linspace(plot_df["age"].min(), plot_df["age"].max(), 200)
        x_range_centered = x_range_raw - baseline_mean_age
        y_pred = fixed_intercept + fixed_slope * x_range_centered

        # Standard Error of fixed population line using the analytical covariance matrix parameters
        X_mat = np.column_stack([np.ones_like(x_range_centered), x_range_centered])
        cov_fe = result.cov_params().loc[['Intercept', 'age_centered'], ['Intercept', 'age_centered']].values
        se_fit = np.sqrt(np.diag(np.dot(np.dot(X_mat, cov_fe), X_mat.T)))
        y_lo = y_pred - 1.96 * se_fit
        y_hi = y_pred + 1.96 * se_fit

        ax.plot(x_range_raw, y_pred, color=color, linewidth=2.5, zorder=4,
                label=f"LMM Fixed Trend (Δ={fixed_slope:+.2f}/yr)")
        ax.fill_between(x_range_raw, y_lo, y_hi, alpha=0.25, color="#e0e0e0", zorder=3)
        
    except Exception as e:
        # Emergency fallback to pooled OLS visually if mathematical calculation breaks
        fit = ols_with_ci(plot_df["age"].values, plot_df[col].values)
        if fit:
            x_range, y_pred, y_lo, y_hi, r, p, slope = fit
            ax.plot(x_range, y_pred, color=color, linewidth=2.5, zorder=4, label=f"OLS: slope={slope:.2f}")
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
# FIGURE 2 – Baseline score vs individual rate of change
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 2: Baseline vs slope …")

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle(
    "Baseline Cognitive Score vs Individual Rate of Change\n"
    "Each dot = one participant. Slope = LMM random slope (score units / year) "
    "from their personal trajectory model.\n"
    "If r ≈ 0: baseline and slope carry independent information → slope adds ML value.",
    fontsize=11, y=1.02,
)
axes2_flat = axes2.flatten()
axes2_flat[5].set_visible(False)

r_summary = {}

for ax_idx, (domain_key, domain) in enumerate(DOMAINS.items()):
    ax    = axes2_flat[ax_idx]
    col   = domain["col"]
    color = DOMAIN_COLORS[domain_key]

    data = df.copy()
    if domain["filter"]:
        fcol, fval = domain["filter"]
        data = data[data[fcol] == fval]

    bs_df = compute_baseline_slope(data, col)
    bs_df = bs_df.dropna(subset=["baseline", "slope"])

    if len(bs_df) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        continue

    ax.scatter(
        bs_df["baseline"], bs_df["slope"],
        color=color, alpha=0.35, s=18, linewidths=0, zorder=2,
    )

    fit = ols_with_ci(bs_df["baseline"].values, bs_df["slope"].values)
    if fit:
        x_range, y_pred, y_lo, y_hi, r, p, _ = fit
        ax.plot(x_range, y_pred, color=color, linewidth=2, zorder=3)
        ax.fill_between(x_range, y_lo, y_hi, alpha=0.18, color=color, zorder=2)
        r_summary[domain_key] = r
    else:
        r, p = stats.pearsonr(bs_df["baseline"], bs_df["slope"])
        r_summary[domain_key] = r

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    p_label   = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    interp    = (
        "Baseline ≈ independent of slope"    if abs(r) < 0.15 else
        "Weak baseline–slope association"    if abs(r) < 0.30 else
        "Moderate baseline–slope association" if abs(r) < 0.50 else
        "Strong baseline–slope association"
    )
    ax.text(
        0.04, 0.97,
        f"r = {r:.2f}   {p_label}\nn = {len(bs_df)}\n{interp}",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    ax.set_xlabel(f"Model Baseline score (Centered intercept)\n{domain['ylabel']}", labelpad=4)
    ax.set_ylabel("Individual LMM slope\n(score units / year)", labelpad=4)
    ax.set_title(domain["title"], fontweight="bold", pad=8)

ax_bar = axes2_flat[5]
ax_bar.set_visible(True)
if r_summary:
    domain_labels = [d.replace("RespInhib", "Resp.\nInhib.")
                       .replace("WorkingMem", "Working\nMem.")
                     for d in r_summary.keys()]
    r_vals  = list(r_summary.values())
    colours = [DOMAIN_COLORS[d] for d in r_summary.keys()]
    bars = ax_bar.bar(domain_labels, r_vals, color=colours, edgecolor="white",
                      linewidth=0.8, zorder=2)
    ax_bar.axhline(0, color="#555555", linewidth=0.8, zorder=1)
    ax_bar.axhline( 0.3, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=1)
    ax_bar.axhline(-0.3, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=1)
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_ylabel("Pearson r (baseline vs slope)", labelpad=4)
    ax_bar.set_title("Baseline–Slope Correlation\nSummary", fontweight="bold", pad=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.text(
        0.5, -0.18,
        "Dotted lines: |r| = 0.3 threshold",
        transform=ax_bar.transAxes, ha="center", fontsize=7.5, color="#777777",
    )
    for bar, r_val in zip(bars, r_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            r_val + (0.03 if r_val >= 0 else -0.06),
            f"{r_val:.2f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "baseline_vs_slope.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Saved → {out2}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Baseline–slope Pearson r summary (LMM Extracted)")
print("=" * 60)
print(f"{'Domain':<22}  {'r':>6}  {'Interpretation'}")
print("-" * 60)
for domain_key, r in r_summary.items():
    interp = (
        "baseline & slope largely independent"  if abs(r) < 0.15 else
        "weak association"                       if abs(r) < 0.30 else
        "moderate association"                   if abs(r) < 0.50 else
        "strong association"
    )
    print(f"  {domain_key:<20}  {r:>6.3f}  {interp}")

print("\nML implication:")
print("  |r| < ~0.30 → slope adds independent predictive power beyond baseline.")
print("  |r| > ~0.50 → baseline likely captures most of the trajectory information.")
print("=" * 60)
print("\nDone.")
