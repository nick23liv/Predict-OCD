"""
visualise_cognition_LMM.py
==========================
Produces two figures exploring longitudinal raw cognitive trajectories.

TIERED (master doc §2.4) — RUN ONCE PER TIER, every output filename is suffixed:

  --tier ml    the selected metric per domain (5 for ABCD). Reads
               harmonised_cognition_ABCDv7_ml.csv.
  --tier all   additionally the viable alternatives and the evaluated-but-
               REJECTED metrics (7 for ABCD). Reads
               QC/evaluated_metrics_ABCDv7.csv, the only file carrying the
               rejected metrics, falling back to the broad CSV if nothing was
               rejected. Slow: Figure 2 fits every metric x two mixed models.

  Figure 1 (trajectories_ABCDv7_<tier>.png)
    - Individual spaghetti lines (score vs age) per metric.
    - Population LMM trend + 95 % CI overlaid (random slope, REML).
    - Points coloured by wave so the cross-sectional age shift is visible
      alongside the within-person trajectory.
    - One shared x-range across all panels (AGE_XLIM), never per-panel scaling.

  Figure 2 (same_vs_individual_slopes_ABCDv7_<tier>.png)
    - Model comparison: does everyone change at the same rate (Model A),
      or do participants differ in their rate of change (Model B)?
    - Three columns per metric: Model A trajectories (parallel), Model B
      trajectories (fanning), and a statistics box (dBIC, LRT, R2, SD).
    - Fitted by ML (not REML), optimiser `powell`, boundary-corrected LRT.
    - dBIC is SIGNED: positive favours Model B, negative favours Model A,
      anything within +/-10 is a tie regardless of sign.
    - Requires >= 3 observations per PARTICIPANT; with 2 the line fits exactly
      and slope variance is confounded with residual variance.

Panels are named by TASK where a domain carries alternatives, and never mark
which metric was selected — the reader judges from the data.

Usage
-----
  python visualise_cognition_LMM.py --tier ml
  python visualise_cognition_LMM.py --tier all

  python visualise_cognition_LMM.py \
      --tier    all \
      --input   /path/to/QC/evaluated_metrics_ABCDv7.csv \
      --dyn     /dataset/abcd/v7/phenotype/ab_g_dyn.tsv \
      --out-dir /path/to/output/figures

Raw variables plotted (inputs to the z-scoring step):
  IQ (raw)                  -> IQ_raw_notz              WISC-V Matrix Reasoning, items correct
  IQ (scaled)        [all]  -> IQ_scaled_notz           WISC-V Matrix Reasoning, scaled score
  Response Inhibition       -> RespInhib_computed_notz  NIH Toolbox Flanker computed score
  Response Inhibition[eval] -> RespInhib_incongr_acc_notz  Millisecond Flanker incongruent accuracy
  Cognitive Flexibility     -> CogFlex_computed_notz    NIH Toolbox DCCS computed score
  Working Memory            -> WorkingMem_raw_notz      List Sorting raw score
  Attention                 -> Attention_raw_notz       Little Man Task accuracy
"""

import argparse
import os
import sys
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
    default=None,
    help="Path to the harmonised cognition CSV. Default depends on --tier: the "
         "ML CSV for 'ml', and the QC evaluation working file for 'all' (which "
         "is the only file carrying the rejected metrics), falling back to the "
         "broad CSV if nothing was rejected.",
)
parser.add_argument(
    "--tier",
    choices=["all", "ml"],
    default="ml",
    help="Which metric set to plot (master doc §2.4). 'ml' plots the selected "
         "metric per domain. 'all' additionally plots the viable alternatives "
         "and the evaluated-but-rejected metrics. RUN ONCE PER TIER — every "
         "output filename is suffixed. Expect 'all' to be slow: Figure 2 fits "
         "every metric x two mixed models.",
)
parser.add_argument(
    "--no-report",
    action="store_true",
    help="Do not mirror stdout to trajectory_report_ABCDv7_<tier>.txt.",
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

TIER = args.tier

# Resolve paths. The 'all' tier reads the QC evaluation working file, because the
# rejected metrics it must plot deliberately never reach a harmonised CSV. Where
# nothing was rejected that file is not written at all, so fall back to the broad
# CSV rather than dying on a missing path.
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()

if args.input:
    INPUT_FILE = args.input
elif TIER == "ml":
    INPUT_FILE = os.path.join(_here, "harmonised_cognition_ABCDv7_ml.csv")
else:
    _eval_file = os.path.join(_here, "QC", "evaluated_metrics_ABCDv7.csv")
    INPUT_FILE = (_eval_file if os.path.exists(_eval_file)
                  else os.path.join(_here, "harmonised_cognition_ABCDv7_all.csv"))

if not os.path.exists(INPUT_FILE):
    sys.exit(
        f"ERROR: input not found: {INPUT_FILE}\n"
        f"Run the harmonisation script for this tier first:\n"
        f"    python harmonising_cognition_v4_ABCDv7.py --tier {TIER}")

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


# ── Run transcript ─────────────────────────────────────────────────────────────
# This one matters more than the harmonisation script's: it holds the ΔBIC
# verdict table, which costs minutes of model fitting to regenerate.
class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


_tee = None
if not args.no_report:
    REPORT_FILE = os.path.join(OUT_DIR, f"trajectory_report_ABCDv7_{TIER}.txt")
    _tee = _Tee(sys.stdout, REPORT_FILE)
    sys.stdout = _tee
    print(f"Run transcript: {REPORT_FILE}")

print(f"Tier: {TIER}")
print(f"Input: {INPUT_FILE}")

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
# Panel headings are two lines: `name` (bold) then `source` ("task; variable").
# `note` is small grey in-panel text, not part of the heading.
# `waves` optionally restricts a domain to the sessions where it was actually
# administered — see the IQ entry.
METRICS = {
    "IQ_raw": {
        "tier":   "ml",
        "domain": "IQ",
        "domain_name": "IQ",
        "task":   "raw",
        # Raw, not scaled. The raw item count is the variable carried into the
        # pooled models: IMAGEN's WISC-IV Matrix Reasoning has no age-normed
        # column, so raw-to-raw is the only like-for-like ABCD<->IMAGEN link.
        # Because raw is not age-normed, the cross-sectional trend slopes upward
        # across the baseline window where the scaled score is flat.
        "col":    "IQ_raw_notz",
        "source": "WISC-V Matrix Reasoning; raw score",
        "ylabel": "Raw score (items correct)",
        "note":   "32 items; not age-normed",
        "filter": None,
        # WISC-V is a baseline-only measure in ABCD v7: 11,613 observations for
        # 11,611 participants, i.e. a handful of stray rows carry a later
        # session_id. Those are excluded here (with a console note listing
        # exactly what was dropped, so the exclusion stays auditable).
        "waves":  ["ses-00A"],
    },
    "IQ_scaled": {
        "tier":   "all",
        "domain": "IQ",
        "domain_name": "IQ",
        "task":   "scaled",
        "col":    "IQ_scaled_notz",
        "source": "WISC-V Matrix Reasoning; scaled score",
        "ylabel": "Scaled score",
        "note":   "M=10, SD=3 normative scale",
        "filter": None,
        "waves":  ["ses-00A"],
    },
    "RespInhib_computed": {
        "tier":   "ml",
        "domain": "RespInhib",
        "domain_name": "Response Inhibition",
        "task":   "NIH Toolbox",
        "col":    "RespInhib_computed_notz",
        "source": "NIH Toolbox Flanker; computed score",
        "ylabel": "Computed score",
        "note":   "Accuracy + reaction time",
        "filter": None,
        "waves":  None,
    },
    "RespInhib_incongr": {
        # Tier 'eval' — reaches the _all figures only, never a CSV. Presented on
        # the same footing as every other metric: the panel is named by its task
        # and its note stays descriptive, so the reader judges from the data
        # rather than from a label (conversion guide §6).
        "tier":   "eval",
        "domain": "RespInhib",
        "domain_name": "Response Inhibition",
        "task":   "Millisecond",
        "col":    "RespInhib_incongr_acc_notz",
        "source": "Millisecond Flanker; incongruent accuracy",
        "ylabel": "Proportion correct",
        "note":   "Range 0–1; not administered at baseline",
        "filter": None,
        "waves":  None,
    },
    "CogFlex_computed": {
        "tier":   "ml",
        "domain": "CogFlex",
        "domain_name": "Cognitive Flexibility",
        "task":   "NIH Toolbox",
        "col":    "CogFlex_computed_notz",
        "source": "NIH Toolbox DCCS; computed score",
        "ylabel": "Computed score",
        "note":   "",
        "filter": None,
        "waves":  None,
    },
    "WorkingMem_listsort": {
        "tier":   "ml",
        "domain": "WorkingMem",
        "domain_name": "Working Memory",
        "task":   "List Sorting",
        "col":    "WorkingMem_raw_notz",
        "source": "NIH Toolbox List Sorting; raw score",
        "ylabel": "Raw score",
        "note":   "",
        "filter": None,
        "waves":  None,
    },
    "Attention_lmt": {
        "tier":   "ml",
        "domain": "Attention",
        "domain_name": "Attention",
        "task":   "Little Man",
        "col":    "Attention_raw_notz",
        "source": "Little Man Task; accuracy",
        "ylabel": "Proportion correct",
        "note":   "Range 0–1",
        "filter": None,
        "waves":  None,
    },
}

# ── Which metrics this tier plots ─────────────────────────────────────────────
# 'ml' shows the selected metric per domain. 'all' additionally shows the viable
# alternatives AND the metrics that were evaluated and rejected — the rejected
# ones are the point: the figures are the evidence for the rejection.
_ACTIVE = {"ml": {"ml"}, "all": {"ml", "all", "eval"}}[TIER]
DOMAINS = {k: dict(v) for k, v in METRICS.items() if v["tier"] in _ACTIVE}

# Panel headings carry the task only where a domain holds alternatives, so the
# figure reads as a comparison between them. Never mark which one was selected.
for _k, _v in DOMAINS.items():
    _n = sum(1 for _o in DOMAINS.values() if _o["domain"] == _v["domain"])
    _v["name"] = (_v["domain_name"] if _n == 1
                  else f"{_v['domain_name']} ({_v['task']})")

print(f"Tier '{TIER}': plotting {len(DOMAINS)} metric(s) — "
      + ", ".join(DOMAINS))

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

# One shared x-range across all panels. Sizing each panel to its own data draws
# every domain on a different age scale, which makes the wave clusters impossible
# to compare between domains; a single axis lets them line up vertically so
# differences in wave coverage read straight off the figure.
# (Not sharex=True: that suppresses the tick labels on the upper row while
# leaving its axis label in place.)
_measure_cols = [d["col"] for d in DOMAINS.values() if d["col"] in df.columns]
_ages = df.loc[df[_measure_cols].notna().any(axis=1), "age"].dropna()
AGE_XLIM = (_ages.min() - 0.4, _ages.max() + 0.4)
print(f"  shared age axis: {AGE_XLIM[0]:.1f} to {AGE_XLIM[1]:.1f} years")

# Grid derived from the active metric count, not hard-coded: the 'all' tier
# carries more panels than 'ml' (ABCD: 7 vs 5), and one spare cell is reserved
# for the shared legend.
_N_PANELS = len(DOMAINS)
_NCOLS    = 3
_NROWS    = -(-(_N_PANELS + 1) // _NCOLS)          # ceil, +1 for the legend cell
_LEGEND_SLOT = _N_PANELS                            # first cell after the panels

fig1, axes1 = plt.subplots(_NROWS, _NCOLS, figsize=(18, 5 * _NROWS))
fig1.suptitle(
    "Longitudinal Cognitive Trajectories (raw scores)\n"
    "Grey lines = individual participants; coloured line = population LMM fixed trend ± 95 % CI; "  # <-- Updated Title text
    "dots coloured by wave",
    fontsize=12, y=1.01,
)
axes1_flat = axes1.flatten()
for _spare in range(_N_PANELS, len(axes1_flat)):
    axes1_flat[_spare].set_visible(False)

for ax_idx, (domain_key, domain) in enumerate(DOMAINS.items()):
    ax    = axes1_flat[ax_idx]
    col   = domain["col"]
    # Colour by DOMAIN, not by metric key (conversion guide bug 2), so
    # alternative metrics for one construct match and the figure reads as a
    # comparison. A lookup on the metric key would miss and fall through to
    # grey, which renders fine and only shows on inspection.
    color = DOMAIN_COLORS[domain["domain"]]

    data = df.copy()
    if domain["filter"]:
        fcol, fval = domain["filter"]
        data = data[data[fcol] == fval]

    # Restrict to the sessions where this measure was actually administered,
    # reporting exactly what is dropped so the exclusion can be checked against
    # the source data rather than taken on trust.
    if domain.get("waves"):
        _excl = data[~data["session_id"].isin(domain["waves"]) & data[col].notna()]
        if len(_excl):
            _by_wave = _excl["session_id"].value_counts().sort_index()
            _detail = ", ".join(f"{w}: {n}" for w, n in _by_wave.items())
            print(f"  [{domain_key}] restricted to {'/'.join(domain['waves'])} — "
                  f"excluded {len(_excl)} observation(s) at other waves ({_detail})")
        data = data[data["session_id"].isin(domain["waves"])]

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
    # Heading: domain name (bold), with the task and variable on a lighter
    # second line. `pad` reserves the room the second line sits in.
    ax.set_title(domain["name"], fontweight="bold", fontsize=12, pad=20)
    ax.text(0.5, 1.012, domain["source"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#444444")
    # Shared across panels. The trend line above is still drawn only over each
    # domain's own observed range, so nothing is extrapolated into ages where
    # that domain was never collected.
    ax.set_xlim(*AGE_XLIM)
    _info = f"n={n_subj} participants\n{n_obs} observations"
    if domain["note"]:
        _info += f"\n{domain['note']}"
    ax.text(
        0.02, 0.97, _info,
        transform=ax.transAxes, fontsize=8, va="top", color="#555555",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)

legend_ax = axes1_flat[_LEGEND_SLOT]
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
out1 = os.path.join(OUT_DIR, f"trajectories_ABCDv7_{TIER}.png")
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

# A random-slope model needs 3+ observations per person to separate slope
# variance from residual variance; with 2 they are confounded. Kept in step with
# MIN_WAVES_FOR_SLOPES in visualise_cognition_IMAGEN.py.
MIN_OBS_FOR_SLOPES     = 3
MIN_SUBJECTS_FOR_SLOPES = 1000

# IQ has too few repeated measures in ABCD to fit meaningful random slopes.
SLOPE_DOMAINS = [k for k, s in DOMAINS.items() if s["domain"] != "IQ"]

fits2 = {}
for dom in SLOPE_DOMAINS:
    col = DOMAINS[dom]["col"]
    domain_filter = DOMAINS[dom].get("filter")

    d = df.copy()
    if domain_filter:
        fcol, fval = domain_filter
        d = d[d[fcol] == fval]

    # A random-slope model needs at least THREE observations per person. With
    # two, the participant's two points define a line exactly, leaving no
    # residual — so slope variance and residual variance are confounded and
    # Model B "wins" by absorbing noise into the random slope. Cognitive
    # Flexibility is administered at only two waves in ABCD v7 (exactly 2.00
    # observations per eligible participant), and under the previous `>= 2`
    # threshold it reported a spurious ΔBIC of +99 on a degenerate fit.
    d = d.dropna(subset=[col, "age"]).copy()
    cnt = d.groupby("participant_id").size()
    n_2wave = int((cnt >= 2).sum())
    d = d[d["participant_id"].isin(cnt[cnt >= MIN_OBS_FOR_SLOPES].index)]
    n_eligible = d["participant_id"].nunique()

    if n_eligible < MIN_SUBJECTS_FOR_SLOPES:
        print(f"    [skip] {dom}: {n_eligible} participants with "
              f"≥{MIN_OBS_FOR_SLOPES} waves (need {MIN_SUBJECTS_FOR_SLOPES}); "
              f"{n_2wave} have ≥2 — a random slope is not identifiable from "
              f"two timepoints, so this domain cannot be assessed.")
        continue

    print(f"    -> {dom} …", flush=True)
    d["t0"]   = d.groupby("participant_id")["age"].transform("min")
    d["time"] = d["age"] - d["t0"]

    # powell is used for every domain so the optimiser is uniform across the
    # whole figure. It is derivative-free, so unlike lbfgs it never inverts the
    # Hessian and cannot fail with LinAlgError: Singular matrix on a flat
    # likelihood surface (which lbfgs did for RespInhib's Model B). Slower, but
    # the runtime is acceptable and uniformity keeps the methods reporting simple.
    try:
        mA = smf.mixedlm(f"{col} ~ time", d, groups="participant_id",
                         re_formula="~1").fit(reml=False, maxiter=300, method="powell")
        mB = smf.mixedlm(f"{col} ~ time", d, groups="participant_id",
                         re_formula="~time").fit(reml=False, maxiter=300, method="powell")
    except Exception as e:
        print(f"    [skip] {dom}: model fit failed ({e}).")
        continue

    # Powell with a maxiter cap can stop short of a true optimum, which would
    # make the A-vs-B comparison meaningless. Say so rather than reporting a
    # ΔBIC computed from a half-converged fit.
    for _name, _m in (("A", mA), ("B", mB)):
        if not getattr(_m, "converged", True):
            print(f"    [WARN] {dom}: Model {_name} did NOT converge — "
                  f"treat the comparison below as unreliable.")

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
            ttl = ("Model A — ONE shared rate of change" if mdl == "A"
                   else "Model B — INDIVIDUAL rates of change")
            ax.set_title(f"{DOMAINS[dom]['name']}\n{ttl}", fontsize=10, fontweight="bold")
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
        # One colour per verdict, so the box never says "Model B better" while
        # looking identical to an inconclusive one. Greens favour individual
        # rates, oranges favour a single shared rate, grey is a genuine tie;
        # the deeper shade is the >50 "very strong" tier in each direction.
        _edge = ("#2ca02c" if _db >  50 else      # green        very strong, B
                 "#7cb342" if _db >  10 else      # light green  strong, B
                 "#cccccc" if _db > -10 else      # grey         inconclusive
                 "#f0a860" if _db > -50 else      # light orange strong, A
                 "#d95f02")                       # orange       very strong, A
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
    out2 = os.path.join(OUT_DIR, f"same_vs_individual_slopes_ABCDv7_{TIER}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {out2}")

    print("\n" + "=" * 74)
    print("SAME RATE FOR EVERYONE (A)  vs  INDIVIDUAL RATES (B)")
    print("=" * 74)
    print(f"{'Metric':<22}{'subj':>6}{'dBIC':>8}{'LRT':>8}{'p':>10}"
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
        print(f"{dom:<22}{f['n_subj']:>6}{_db:>+8.0f}{f['lrt']:>8.0f}"
              f"{f['p']:>10.0e}{f['r2A']:>8.3f}{f['r2B']:>8.3f}"
              f"{(f['r2B']-f['r2A'])*100:>+7.1f}pp   {v}")
    print("\nΔBIC is signed: POSITIVE favours Model B (individual rates of change),")
    print("NEGATIVE favours Model A (one shared rate). |ΔBIC| > 50 = very strong.")
    print("A significant LRT with a negative ΔBIC means the effect is real but too")
    print("small to justify the extra parameters at this sample size.")
    print("=" * 74)

print("\nDone.")


# ── Close the run transcript ──────────────────────────────────────────────────
if _tee is not None:
    sys.stdout = _tee.stream
    _tee.close()
    print(f"Run transcript written to: {REPORT_FILE}")
