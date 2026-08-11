#!/usr/bin/env python
# coding: utf-8

# harmonising_cognition_v4_ABCDv7.py
# Function: Harmonises ABCD Study (v7) data across 5 cognitive domains.
#
# Z-scoring variants produced per domain
# ----------------------------------------
#   _z_withinsite      : z-scored within each ABCD site
#   _z_withinsitewave  : z-scored within each site × wave combination
#   _z_withincohort    : z-scored across all sites (full cohort as one group)
#   _z_withincohortwave: z-scored across all sites but separately within each wave
#
# Changelog
# ---------
#   v1: within-site and within-site × wave z-scoring for all 5 domains
#   v2: added _withincohort and _withincohortwave z-scoring for all 5 domains;
#       Millisecond Flanker used mean(acc_z, −RT_z)
#   v3: Millisecond Flanker z-score changed to accuracy only (overall acc);
#       RespInhib_medrt_notz removed from output;
#       fixed column names: nc_y_nihtb__flnkr__computed_score,
#                           nc_y_nihtb__crdst__computed_score
#   v4: Millisecond Flanker z-score uses incongruent accuracy only
#       (nc_y_flnkr__incongr_acc), not overall accuracy — incongruent trials
#       are the direct measure of conflict resolution and are consistent with
#       what the NIH Toolbox Flanker computed_score is built around;
#       output column renamed RespInhib_acc_notz → RespInhib_incongr_acc_notz
#
# Output: harmonised_cognition_v4_ABCDv7.csv


import argparse
import os
import pandas as pd
import numpy as np


# ── CLI arguments ──────────────────────────────────────────────────────────────
try:
    default_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    default_dir = os.getcwd()

parser = argparse.ArgumentParser(description="Harmonise ABCD cognitive data (v4).")
parser.add_argument(
    "--data-dir",
    default=default_dir,
    help="Directory containing the input TSV files.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Full path for the output CSV. "
         "Default: harmonised_cognition_v4_ABCDv7.csv inside --data-dir.",
)
parser.add_argument(
    "--skip-qc",
    action="store_true",
    help="Skip the post-harmonisation QC report and figures.",
)
parser.add_argument(
    "--qc-dir",
    default=None,
    help="Directory for QC figures. Default: a 'QC' subfolder inside --data-dir.",
)

try:
    __file__
    args = parser.parse_args()
except NameError:
    args = parser.parse_args(args=[])

DATA_DIR = args.data_dir

FILES = {
    "dyn":   os.path.join(DATA_DIR, "ab_g_dyn.tsv"),
    "wisc":  os.path.join(DATA_DIR, "nc_y_wisc.tsv"),
    "flnkr": os.path.join(DATA_DIR, "nc_y_flnkr.tsv"),
    "nihtb": os.path.join(DATA_DIR, "nc_y_nihtb.tsv"),
    "lmt":   os.path.join(DATA_DIR, "nc_y_lmt.tsv"),
}

OUT_FILE = (
    args.output
    if args.output
    else os.path.join(DATA_DIR, "harmonised_cognition_v4_ABCDv7.csv")
)


# ── Utility: robust within-group z-score ──────────────────────────────────────
def zscore_within_groups(series: pd.Series, groups: list) -> pd.Series:
    """
    Z-score `series` within each combination of `groups`.
    Groups with n ≤ 1 or SD = 0 are set to 0 (the group mean) rather than NaN.

    To z-score across the full cohort with no grouping, pass a constant Series:
        zscore_within_groups(series, [pd.Series("all", index=series.index)])
    """
    def _safe_z(x):
        n   = x.notna().sum()
        std = x.std(ddof=1)
        if n <= 1 or std == 0 or pd.isna(std):
            return pd.Series(0.0, index=x.index).where(x.notna(), other=np.nan)
        return (x - x.mean()) / std

    result = pd.Series(np.nan, index=series.index)
    if isinstance(groups, pd.Series):
        groups = [groups]

    key = groups[0]
    for i in range(1, len(groups)):
        key = key.astype(str) + "__" + groups[i].astype(str)

    for _, idx in series.groupby(key, sort=False).groups.items():
        result.iloc[result.index.get_indexer(idx)] = _safe_z(series.loc[idx]).values

    return result


def _cohort_key(index):
    """Returns a constant Series so zscore_within_groups treats all rows as one group."""
    return pd.Series("all", index=index)


# Domain metadata, used by the QC section at the end of this script.
#   raw        : the raw (_notz) column name
#   flipped    : True if z-score was sign-flipped (higher raw = worse → higher z = better)
#   plausible  : (min, max) plausible range for the RAW score; values outside
#                are flagged as likely data errors, not just outliers
#   bounds     : (min, max) hard scale bounds for floor/ceiling check; None = continuous
#   label/units: for plot axes
#   age_expect : expected sign of r(raw, age) — +1 positive, -1 negative, 0 ≈ zero (normed)
#   z_prefix   : prefix for z-score column names (e.g. "CogFlex" → CogFlex_z_withinsite)
#
# Note: Response Inhibition has two raw sub-columns (NIH Toolbox Flanker and Millisecond
# Flanker) that are z-scored separately and then merged into combined RespInhib_z_* columns.
# Both sub-columns are listed here for raw checks; z-score checks use the shared z_prefix.
DOMAIN_SPEC = {
    "IQ_scaled": {
        "raw":       "IQ_scaled_notz",
        "flipped":   False,
        "plausible": (1, 19),
        "bounds":    (1, 19),
        "label":     "IQ (WISC-V scaled score)",
        "units":     "scaled score (M=10, SD=3)",
        "age_expect": 0,
        "z_prefix":  "IQ_scaled",
    },
    "IQ_raw": {
        "raw":       "IQ_raw_notz",
        "flipped":   False,
        "plausible": (0, 30),
        "bounds":    None,
        "label":     "IQ (WISC-V raw score)",
        "units":     "raw score",
        "age_expect": +1,
        "z_prefix":  "IQ_raw",
    },
    "RespInhib_computed": {
        "raw":       "RespInhib_computed_notz",
        "flipped":   False,
        "plausible": (0, 150),
        "bounds":    None,
        "label":     "Response Inhibition (NIH Toolbox Flanker computed score)",
        "units":     "computed score",
        "age_expect": +1,
        "z_prefix":  "RespInhib",
    },
    "RespInhib_incongr_acc": {
        "raw":       "RespInhib_incongr_acc_notz",
        "flipped":   False,
        "plausible": (0, 1),
        "bounds":    (0.0, 1.0),
        "label":     "Response Inhibition (Millisecond Flanker incongruent accuracy)",
        "units":     "proportion correct",
        "age_expect": +1,
        "z_prefix":  "RespInhib",
    },
    "CogFlex": {
        "raw":       "CogFlex_computed_notz",
        "flipped":   False,
        "plausible": (0, 150),
        "bounds":    None,
        "label":     "Cognitive Flexibility (DCCS computed score)",
        "units":     "computed score",
        "age_expect": +1,
        "z_prefix":  "CogFlex",
    },
    "WorkingMem": {
        "raw":       "WorkingMem_raw_notz",
        "flipped":   False,
        "plausible": (0, 26),
        "bounds":    (0, 26),
        "label":     "Working Memory (List Sorting raw score)",
        "units":     "raw score",
        "age_expect": +1,
        "z_prefix":  "WorkingMem",
    },
    "Attention": {
        "raw":       "Attention_raw_notz",
        "flipped":   False,
        "plausible": (0, 1),
        "bounds":    (0.0, 1.0),
        "label":     "Attention (Little Man Task proportion correct)",
        "units":     "proportion correct",
        "age_expect": +1,
        "z_prefix":  "Attention",
    },
}

# Deduplicated view used for z-score checks (checks 4, 5, 6, 10, 12).
# RespInhib_computed and RespInhib_incongr_acc share the same combined z columns,
# so only RespInhib_computed appears here as the representative entry.
_seen_z: set = set()
Z_SPEC: dict = {}
for _dom, _spec in DOMAIN_SPEC.items():
    if _spec["z_prefix"] not in _seen_z:
        Z_SPEC[_dom] = _spec
        _seen_z.add(_spec["z_prefix"])


# ── Load files ─────────────────────────────────────────────────────────────────
print("Loading data files …")
dyn   = pd.read_csv(FILES["dyn"],   sep="\t", na_values="n/a", low_memory=False)
wisc  = pd.read_csv(FILES["wisc"],  sep="\t", na_values="n/a", low_memory=False)
flnkr = pd.read_csv(FILES["flnkr"], sep="\t", na_values="n/a", low_memory=False)
nihtb = pd.read_csv(FILES["nihtb"], sep="\t", na_values="n/a", low_memory=False)
lmt   = pd.read_csv(FILES["lmt"],   sep="\t", na_values="n/a", low_memory=False)


# ── Extract site (one row per participant × session) ───────────────────────────
print("Extracting site information …")
site_df = (
    dyn[["participant_id", "session_id", "ab_g_dyn__design_site"]]
    .dropna(subset=["ab_g_dyn__design_site"])
    .drop_duplicates(subset=["participant_id", "session_id"], keep="first")
    .rename(columns={"ab_g_dyn__design_site": "site"})
)


# ── Build master spine ─────────────────────────────────────────────────────────
print("Building master participant × session spine …")
all_ids = pd.concat([
    wisc [["participant_id", "session_id"]],
    flnkr[["participant_id", "session_id"]],
    nihtb[["participant_id", "session_id"]],
    lmt  [["participant_id", "session_id"]],
    site_df[["participant_id", "session_id"]],
]).drop_duplicates()

master = all_ids.merge(site_df, on=["participant_id", "session_id"], how="left")


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 – IQ  (WISC-V Matrix Reasoning)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 1: IQ …")

iq = wisc[["participant_id", "session_id",
           "nc_y_wisc__raw_score", "nc_y_wisc__scaled_score"]].copy()
iq = iq.rename(columns={
    "nc_y_wisc__raw_score":    "IQ_raw_notz",
    "nc_y_wisc__scaled_score": "IQ_scaled_notz",
})
iq = iq.merge(site_df, on=["participant_id", "session_id"], how="left")

iq["IQ_raw_z_withinsite"]    = zscore_within_groups(iq["IQ_raw_notz"],    [iq["site"]])
iq["IQ_scaled_z_withinsite"] = zscore_within_groups(iq["IQ_scaled_notz"], [iq["site"]])

iq["IQ_raw_z_withinsitewave"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [iq["site"], iq["session_id"]])
iq["IQ_scaled_z_withinsitewave"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [iq["site"], iq["session_id"]])

iq["IQ_raw_z_withincohort"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [_cohort_key(iq.index)])
iq["IQ_scaled_z_withincohort"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [_cohort_key(iq.index)])

iq["IQ_raw_z_withincohortwave"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [iq["session_id"]])
iq["IQ_scaled_z_withincohortwave"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [iq["session_id"]])

iq_out = iq[["participant_id", "session_id",
             "IQ_raw_notz", "IQ_scaled_notz",
             "IQ_raw_z_withinsite",       "IQ_scaled_z_withinsite",
             "IQ_raw_z_withinsitewave",   "IQ_scaled_z_withinsitewave",
             "IQ_raw_z_withincohort",     "IQ_scaled_z_withincohort",
             "IQ_raw_z_withincohortwave", "IQ_scaled_z_withincohortwave"]]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 – Response Inhibition
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 2: Response Inhibition …")

# 2a. NIH Toolbox Flanker
# computed_score uses a two-stage algorithm:
#   - incongruent accuracy ≤ 80%: score is accuracy-based
#   - incongruent accuracy > 80%: score is RT-cost-based (congruent RT − incongruent RT)
# Higher computed_score = better inhibitory control. No flip applied.
nihtb_flnkr = nihtb[["participant_id", "session_id",
                       "nc_y_nihtb__flnkr__computed_score"]].copy()
nihtb_flnkr = nihtb_flnkr.rename(
    columns={"nc_y_nihtb__flnkr__computed_score": "nihtb_computed_score"})

# 2b. Millisecond Flanker — incongruent accuracy only (v4)
# Incongruent trials are the direct measure of conflict resolution.
# This is consistent with what the NIH Toolbox Flanker computed_score is built around.
# Overall accuracy (v3) was replaced because congruent trials do not test response inhibition.
ms_flnkr = flnkr[["participant_id", "session_id",
                    "nc_y_flnkr__incongr_acc"]].copy()
ms_flnkr = ms_flnkr.rename(columns={
    "nc_y_flnkr__incongr_acc": "ms_incongr_acc",
})

# 2c. Merge and assign task_type
# Rule: NIH Toolbox Flanker preferred; Millisecond used only when NIH unavailable.
ri = nihtb_flnkr.merge(ms_flnkr, on=["participant_id", "session_id"], how="outer")
ri = ri.merge(site_df,            on=["participant_id", "session_id"], how="left")

ri["task_type"] = pd.NA
ri.loc[ri["nihtb_computed_score"].notna(), "task_type"] = "nihtb_flanker"
ri.loc[
    ri["nihtb_computed_score"].isna() & ri["ms_incongr_acc"].notna(),
    "task_type"
] = "millisecond_flanker"

# 2d. Raw output columns
ri["RespInhib_computed_notz"] = np.where(
    ri["task_type"] == "nihtb_flanker", ri["nihtb_computed_score"], np.nan)
ri["RespInhib_incongr_acc_notz"] = np.where(
    ri["task_type"] == "millisecond_flanker", ri["ms_incongr_acc"], np.nan)

nihtb_mask = ri["task_type"] == "nihtb_flanker"
ms_mask    = ri["task_type"] == "millisecond_flanker"

# ── Z-scoring ─────────────────────────────────────────────────────────────────
# NIH Toolbox Flanker: higher computed_score = better → no flip
# Millisecond Flanker: higher incongruent accuracy = better → no flip
# Z-scoring is computed separately within each task type before combining,
# so the two tasks' different scales do not contaminate each other.

for suffix, site_groups, ms_groups in [
    ("_withinsite",       [ri["site"]],                   [ri["site"]]),
    ("_withinsitewave",   [ri["site"], ri["session_id"]], [ri["site"], ri["session_id"]]),
    ("_withincohort",     [_cohort_key(ri.index)],        [_cohort_key(ri.index)]),
    ("_withincohortwave", [ri["session_id"]],             [ri["session_id"]]),
]:
    ri.loc[nihtb_mask, f"_nihtb_z{suffix}"] = zscore_within_groups(
        ri.loc[nihtb_mask, "nihtb_computed_score"],
        [g.loc[nihtb_mask] for g in site_groups],
    )
    ri.loc[ms_mask, f"_ms_z{suffix}"] = zscore_within_groups(
        ri.loc[ms_mask, "ms_incongr_acc"],
        [g.loc[ms_mask] for g in ms_groups],
    )
    col = f"RespInhib_z{suffix}"
    ri[col] = np.nan
    ri.loc[nihtb_mask, col] = ri.loc[nihtb_mask, f"_nihtb_z{suffix}"]
    ri.loc[ms_mask,    col] = ri.loc[ms_mask,    f"_ms_z{suffix}"]

ri_out = ri[["participant_id", "session_id",
             "RespInhib_computed_notz", "RespInhib_incongr_acc_notz",
             "RespInhib_z_withinsite",       "RespInhib_z_withinsitewave",
             "RespInhib_z_withincohort",     "RespInhib_z_withincohortwave",
             "task_type"]]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 – Cognitive Flexibility  (NIH Toolbox DCCS / Card Sort)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 3: Cognitive Flexibility …")

cf = nihtb[["participant_id", "session_id",
             "nc_y_nihtb__crdst__computed_score"]].copy()
cf = cf.rename(columns={"nc_y_nihtb__crdst__computed_score": "CogFlex_computed_notz"})
cf = cf.merge(site_df, on=["participant_id", "session_id"], how="left")

cf["CogFlex_z_withinsite"]       = zscore_within_groups(
    cf["CogFlex_computed_notz"], [cf["site"]])
cf["CogFlex_z_withinsitewave"]   = zscore_within_groups(
    cf["CogFlex_computed_notz"], [cf["site"], cf["session_id"]])
cf["CogFlex_z_withincohort"]     = zscore_within_groups(
    cf["CogFlex_computed_notz"], [_cohort_key(cf.index)])
cf["CogFlex_z_withincohortwave"] = zscore_within_groups(
    cf["CogFlex_computed_notz"], [cf["session_id"]])

cf_out = cf[["participant_id", "session_id",
             "CogFlex_computed_notz",
             "CogFlex_z_withinsite",       "CogFlex_z_withinsitewave",
             "CogFlex_z_withincohort",     "CogFlex_z_withincohortwave"]]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 4 – Working Memory  (NIH Toolbox List Sorting)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 4: Working Memory …")

wm = nihtb[["participant_id", "session_id",
             "nc_y_nihtb__lswmt__raw_score"]].copy()
wm = wm.rename(columns={"nc_y_nihtb__lswmt__raw_score": "WorkingMem_raw_notz"})
wm = wm.merge(site_df, on=["participant_id", "session_id"], how="left")

wm["WorkingMem_z_withinsite"]       = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [wm["site"]])
wm["WorkingMem_z_withinsitewave"]   = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [wm["site"], wm["session_id"]])
wm["WorkingMem_z_withincohort"]     = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [_cohort_key(wm.index)])
wm["WorkingMem_z_withincohortwave"] = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [wm["session_id"]])

wm_out = wm[["participant_id", "session_id",
             "WorkingMem_raw_notz",
             "WorkingMem_z_withinsite",       "WorkingMem_z_withinsitewave",
             "WorkingMem_z_withincohort",     "WorkingMem_z_withincohortwave"]]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 5 – Attention  (Little Man Task)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 5: Attention …")

att = lmt[["participant_id", "session_id",
            "nc_y_lmt__crct_acc"]].copy()
att = att.rename(columns={"nc_y_lmt__crct_acc": "Attention_raw_notz"})
# A small number of rows have values >1 (coded as percentage rather than proportion).
# These are data quality errors — set them to NaN so they are excluded from z-scoring.
n_bad = (att["Attention_raw_notz"] > 1).sum()
if n_bad > 0:
    print(f"  Attention: excluding {n_bad} rows with nc_y_lmt__crct_acc > 1 (percentage coding error)")
    att.loc[att["Attention_raw_notz"] > 1, "Attention_raw_notz"] = np.nan
att = att.merge(site_df, on=["participant_id", "session_id"], how="left")

att["Attention_z_withinsite"]       = zscore_within_groups(
    att["Attention_raw_notz"], [att["site"]])
att["Attention_z_withinsitewave"]   = zscore_within_groups(
    att["Attention_raw_notz"], [att["site"], att["session_id"]])
att["Attention_z_withincohort"]     = zscore_within_groups(
    att["Attention_raw_notz"], [_cohort_key(att.index)])
att["Attention_z_withincohortwave"] = zscore_within_groups(
    att["Attention_raw_notz"], [att["session_id"]])

att_out = att[["participant_id", "session_id",
               "Attention_raw_notz",
               "Attention_z_withinsite",       "Attention_z_withinsitewave",
               "Attention_z_withincohort",     "Attention_z_withincohortwave"]]


# ══════════════════════════════════════════════════════════════════════════════
# COMBINE ALL DOMAINS
# ══════════════════════════════════════════════════════════════════════════════
print("Combining all domains …")

combined = master.copy()
for df in [iq_out, ri_out, cf_out, wm_out, att_out]:
    combined = combined.merge(df, on=["participant_id", "session_id"], how="left")

combined["task_type"] = combined["task_type"].fillna("n/a")

col_order = [
    "participant_id", "session_id", "site",
    # IQ
    "IQ_raw_notz", "IQ_scaled_notz",
    "IQ_raw_z_withinsite",         "IQ_scaled_z_withinsite",
    "IQ_raw_z_withinsitewave",     "IQ_scaled_z_withinsitewave",
    "IQ_raw_z_withincohort",       "IQ_scaled_z_withincohort",
    "IQ_raw_z_withincohortwave",   "IQ_scaled_z_withincohortwave",
    # Response Inhibition
    "RespInhib_computed_notz", "RespInhib_incongr_acc_notz",
    "RespInhib_z_withinsite",       "RespInhib_z_withinsitewave",
    "RespInhib_z_withincohort",     "RespInhib_z_withincohortwave",
    "task_type",
    # Cognitive Flexibility
    "CogFlex_computed_notz",
    "CogFlex_z_withinsite",         "CogFlex_z_withinsitewave",
    "CogFlex_z_withincohort",       "CogFlex_z_withincohortwave",
    # Working Memory
    "WorkingMem_raw_notz",
    "WorkingMem_z_withinsite",      "WorkingMem_z_withinsitewave",
    "WorkingMem_z_withincohort",    "WorkingMem_z_withincohortwave",
    # Attention
    "Attention_raw_notz",
    "Attention_z_withinsite",       "Attention_z_withinsitewave",
    "Attention_z_withincohort",     "Attention_z_withincohortwave",
]
combined = combined[col_order]

# Keep a numeric copy for QC BEFORE NaN → "n/a" string replacement
# (the replacement coerces every column to dtype=object, breaking numeric checks).
qc_df = combined.copy()

combined = combined.where(combined.notna(), other="n/a")


# ── Save ───────────────────────────────────────────────────────────────────────
combined.to_csv(OUT_FILE, index=False, na_rep="n/a")
print(f"\nDone.  Output saved to: {OUT_FILE}")
print(f"Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
print("\nColumn list:")
for c in combined.columns:
    print(f"  {c}")


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CONTROL
# ══════════════════════════════════════════════════════════════════════════════
# Runs on the numeric `qc_df` (saved before the NaN → "n/a" replacement).
# Every check prints PASS / WARN / FAIL so problems are visible in a log
# without opening the CSV. Three figures are written to --qc-dir.
# Skip with --skip-qc; redirect figures with --qc-dir.
if not args.skip_qc:
    QC_DIR = args.qc_dir if args.qc_dir else os.path.join(DATA_DIR, "QC")
    os.makedirs(QC_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_fail = 0
    n_warn = 0

    def _status(ok, warn=False):
        global n_fail, n_warn
        if ok:
            return "PASS"
        if warn:
            n_warn += 1
            return "WARN"
        n_fail += 1
        return "FAIL"

    def _zcol(dom_key, suffix):
        """Z-column name for a domain given a suffix like '_z_withinsite'."""
        return f"{DOMAIN_SPEC[dom_key]['z_prefix']}{suffix}"

    ALL_Z_SUFFIXES = [
        "_z_withinsite", "_z_withinsitewave",
        "_z_withincohort", "_z_withincohortwave",
    ]

    # Merge visit age from dyn for construct-validity and attrition checks.
    # dyn is already loaded above; reuse it here without an extra file read.
    age_qc = (
        dyn[["participant_id", "session_id", "ab_g_dyn__visit_age"]]
        .dropna(subset=["ab_g_dyn__visit_age"])
        .drop_duplicates(subset=["participant_id", "session_id"], keep="first")
        .rename(columns={"ab_g_dyn__visit_age": "age"})
    )
    qc_df = qc_df.merge(age_qc, on=["participant_id", "session_id"], how="left")

    ALL_WAVES = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
                 "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
    raw_cols_list = [s["raw"] for s in DOMAIN_SPEC.values()]
    DATA_WAVES = [
        w for w in ALL_WAVES
        if w in qc_df["session_id"].values
        and qc_df.loc[qc_df["session_id"] == w, raw_cols_list].notna().any().any()
    ]

    print("\n" + "=" * 78)
    print("QUALITY CONTROL REPORT")
    print("=" * 78)

    # ── QC 1: row uniqueness ──────────────────────────────────────────────────
    # Duplicate rows silently inflate n and distort every group mean/SD.
    print("\n[1] Row uniqueness (participant_id x session_id)")
    n_dup = qc_df.duplicated(subset=["participant_id", "session_id"]).sum()
    print(f"    duplicate rows: {n_dup}   [{_status(n_dup == 0)}]")

    # ── QC 2: missingness per domain x wave ───────────────────────────────────
    # A domain unexpectedly empty at a wave usually means a variable/form mismatch.
    print("\n[2] Non-missing counts per domain x wave")
    miss = qc_df.groupby("session_id")[raw_cols_list].apply(lambda g: g.notna().sum())
    miss.columns = list(DOMAIN_SPEC.keys())
    miss_show = miss[miss.sum(axis=1) > 0]
    print(miss_show.to_string())
    print(f"\n    (total rows: {len(qc_df)}, "
          f"unique participants: {qc_df['participant_id'].nunique()})")

    # ── QC 3: raw plausible-range check ───────────────────────────────────────
    # Catches impossible values — e.g. LMT proportion > 1 (percentage coding error).
    print("\n[3] Raw values within plausible range")
    for dom, spec in DOMAIN_SPEC.items():
        s = qc_df[spec["raw"]].dropna()
        lo, hi = spec["plausible"]
        n_out = int(((s < lo) | (s > hi)).sum())
        rng = f"[{s.min():.3g}, {s.max():.3g}]" if len(s) else "n/a"
        print(f"    {dom:<28} expected [{lo}, {hi}]  observed {rng:<22} "
              f"out-of-range: {n_out}   [{_status(n_out == 0)}]")

    # ── QC 4: z-scores exist exactly where raw exists ─────────────────────────
    # A mismatch means z-scoring dropped or invented observations.
    # RespInhib: z column covers BOTH sub-tasks; raw_n is the union.
    print("\n[4] Z-score coverage matches raw coverage")
    for dom, spec in Z_SPEC.items():
        if dom == "RespInhib_computed":
            raw_n = qc_df[["RespInhib_computed_notz",
                            "RespInhib_incongr_acc_notz"]].notna().any(axis=1).sum()
        else:
            raw_n = qc_df[spec["raw"]].notna().sum()
        for suffix in ALL_Z_SUFFIXES:
            zcol = _zcol(dom, suffix)
            z_n = qc_df[zcol].notna().sum()
            print(f"    {zcol:<46} raw n={raw_n:<6} z n={z_n:<6} "
                  f"[{_status(raw_n == z_n)}]")

    # ── QC 5: z-scores are correctly standardised ─────────────────────────────
    # _withincohort     → mean 0, SD 1 overall.
    # _withinsite       → mean 0, SD 1 within each site.
    # _withincohortwave → mean 0, SD 1 within each wave.
    # _withinsitewave   → mean 0, SD 1 within each site × wave cell.
    # Only sites with unexpected mean/SD are printed for _withinsite to keep
    # output manageable (21 ABCD sites × 7 domains would be very long).
    print("\n[5] Z-score standardisation (expect mean≈0, SD≈1 within the relevant group)")
    TOL_MEAN, TOL_SD = 0.01, 0.01
    for dom, spec in Z_SPEC.items():
        zc = _zcol(dom, "_z_withincohort")
        s = qc_df[zc].dropna()
        if len(s) > 1:
            ok = abs(s.mean()) < TOL_MEAN and abs(s.std(ddof=1) - 1) < TOL_SD
            print(f"    {zc:<46} overall  mean={s.mean():+.4f} SD={s.std(ddof=1):.4f}  [{_status(ok)}]")

    print("\n    per-site failures for _withinsite (passing sites omitted):")
    any_site_fail = False
    for dom, spec in Z_SPEC.items():
        zc = _zcol(dom, "_z_withinsite")
        for site, grp in qc_df.groupby("site"):
            s = grp[zc].dropna()
            if len(s) <= 1:
                continue
            sd = s.std(ddof=1)
            ok = abs(s.mean()) < TOL_MEAN and abs(sd - 1) < TOL_SD
            degenerate = sd == 0
            tag = _status(ok, warn=degenerate) if not ok else None
            if tag is not None:
                print(f"      {zc:<44} site={str(site):<6} n={len(s):<6} "
                      f"mean={s.mean():+.4f} SD={sd:.4f}  [{tag}]")
                any_site_fail = True
    if not any_site_fail:
        print("      (all sites PASS)")

    print("\n    per-wave check for _withincohortwave:")
    for dom, spec in Z_SPEC.items():
        zc = _zcol(dom, "_z_withincohortwave")
        for wave, grp in qc_df.groupby("session_id"):
            s = grp[zc].dropna()
            if len(s) <= 1:
                continue
            sd = s.std(ddof=1)
            ok = abs(s.mean()) < TOL_MEAN and abs(sd - 1) < TOL_SD
            degenerate = sd == 0
            tag = _status(ok, warn=degenerate) if not ok else "PASS"
            print(f"      {zc:<44} {wave:<10} n={len(s):<6} "
                  f"mean={s.mean():+.4f} SD={sd:.4f}  [{tag}]")

    # ── QC 6: direction / sign-flip verification ──────────────────────────────
    # THE important check. Spearman ρ(raw, z) must be +1 (no flip) or -1 (flipped).
    # All ABCD domains are higher = better (no flip), so expect +1 everywhere.
    # For RespInhib the z combines two tasks; check within each task_type subset.
    print("\n[6] Direction check: Spearman rho(raw, z) — no ABCD domains are flipped, expect +1")
    for dom, spec in Z_SPEC.items():
        expected = -1.0 if spec["flipped"] else +1.0
        for suffix in ["_z_withincohort", "_z_withincohortwave"]:
            zcol = _zcol(dom, suffix)
            if suffix == "_z_withincohortwave":
                rhos = []
                for _, grp in qc_df.groupby("session_id"):
                    p = grp[[spec["raw"], zcol]].dropna()
                    if len(p) >= 3 and p[spec["raw"]].nunique() > 1:
                        rhos.append(p[spec["raw"]].corr(p[zcol], method="spearman"))
                rho = np.mean(rhos) if rhos else np.nan
            else:
                pair = qc_df[[spec["raw"], zcol]].dropna()
                rho = pair[spec["raw"]].corr(pair[zcol], method="spearman") if len(pair) >= 3 else np.nan
            ok = np.isfinite(rho) and abs(rho - expected) < 1e-6
            direction = "flipped" if spec["flipped"] else "not flipped"
            print(f"    {zcol:<46} rho={rho:+.4f} (expect {expected:+.0f}, "
                  f"{direction})   [{_status(ok)}]")

    # ── QC 7: z-scoring group sizes ───────────────────────────────────────────
    # Z-scores from very small groups are unstable. Flags site × wave cells with n < 30.
    print("\n[7] Z-scoring group sizes (per site x wave; small n → unstable z)")
    MIN_N = 30
    flagged_any = False
    for dom, spec in DOMAIN_SPEC.items():
        for (site, wave), grp in qc_df.groupby(["site", "session_id"]):
            n = grp[spec["raw"]].notna().sum()
            if 0 < n < MIN_N:
                print(f"    {dom:<28} site={str(site):<6} {wave:<10} n={n:<6} "
                      f"[{_status(False, warn=True)}]")
                flagged_any = True
    if not flagged_any:
        print(f"    (all populated site x wave cells have n ≥ {MIN_N})")

    # ── QC 8: distribution shape ──────────────────────────────────────────────
    # Z-scoring standardises location/scale but does NOT normalise shape.
    # Strong skew survives z-scoring and matters for downstream models.
    print("\n[8] Distribution shape of raw scores (z-scoring does not fix skew)")
    for dom, spec in DOMAIN_SPEC.items():
        s = qc_df[spec["raw"]].dropna()
        if len(s) < 3:
            continue
        sk, ku = s.skew(), s.kurtosis()
        flag = ("" if abs(sk) < 1 else
                ("  <- moderate skew" if abs(sk) < 2 else "  <- strong skew"))
        print(f"    {dom:<28} skew={sk:+.2f}  excess kurtosis={ku:+.2f}{flag}")

    # ── QC 9: floor / ceiling effects ────────────────────────────────────────
    # % of observations at the scale's hard min/max. A pile-up means the task
    # isn't discriminating at that end; z-scoring cannot fix it.
    print("\n[9] Floor / ceiling effects (% of observations at the scale bound)")
    FC_WARN = 15.0
    for dom, spec in DOMAIN_SPEC.items():
        if spec["bounds"] is None:
            print(f"    {dom:<28} (no hard scale bounds -- skipped)")
            continue
        lo, hi = spec["bounds"]
        s = qc_df[spec["raw"]].dropna()
        if s.empty:
            continue
        pct_lo = (s == lo).mean() * 100
        pct_hi = (s == hi).mean() * 100
        worst = max(pct_lo, pct_hi)
        print(f"    {dom:<28} at floor({lo})={pct_lo:5.1f}%  at ceiling({hi})={pct_hi:5.1f}%   "
              f"[{_status(worst < FC_WARN, warn=True)}]")

    # ── QC 10: outliers after z-scoring ──────────────────────────────────────
    print("\n[10] Extreme values after z-scoring (|z| > 4)")
    for dom in Z_SPEC:
        zc = _zcol(dom, "_z_withincohort")
        s = qc_df[zc].dropna()
        if s.empty:
            continue
        n_ext = int((s.abs() > 4).sum())
        pct = n_ext / len(s) * 100
        print(f"    {zc:<46} n={n_ext:<5} ({pct:.3f}%)  max|z|={s.abs().max():.2f}   "
              f"[{_status(pct < 1.0, warn=True)}]")

    # ── QC 11: age association ────────────────────────────────────────────────
    # Each domain should move with age in the expected direction.
    # Flat or reversed = the variable may not measure what we think.
    print("\n[11] Age association of RAW scores (construct-validity check)")
    for dom, spec in DOMAIN_SPEC.items():
        pair = qc_df[[spec["raw"], "age"]].dropna()
        if len(pair) < 10:
            continue
        r = pair[spec["raw"]].corr(pair["age"])
        exp = spec["age_expect"]
        if exp == 0:
            ok = abs(r) < 0.20
            expect_txt = "~0 (age-normed)"
        else:
            ok = np.sign(r) == np.sign(exp) and abs(r) > 0.05
            expect_txt = "negative" if exp < 0 else "positive"
        print(f"    {dom:<28} r(raw, age)={r:+.3f}   expect {expect_txt:<16}  "
              f"[{_status(ok, warn=True)}]")

    # ── QC 12: cross-domain correlations ─────────────────────────────────────
    # All z-scores are sign-aligned so higher = better, so domains should
    # correlate POSITIVELY. A negative correlation implies a sign error.
    print("\n[12] Cross-domain correlations of z-scores (all sign-aligned: higher = better)")
    zcols_corr = {dom: _zcol(dom, "_z_withincohort") for dom in Z_SPEC}
    corr = qc_df[list(zcols_corr.values())].corr()
    corr.index   = list(zcols_corr.keys())
    corr.columns = list(zcols_corr.keys())
    print(corr.round(3).to_string())
    doms_corr = list(zcols_corr.keys())
    for i in range(len(doms_corr)):
        for j in range(i + 1, len(doms_corr)):
            r_val = corr.iloc[i, j]
            if not pd.isna(r_val) and r_val < -0.05:
                print(f"    NEGATIVE correlation {doms_corr[i]} vs {doms_corr[j]}: r={r_val:+.3f}   "
                      f"[{_status(False, warn=True)}]")

    # ── QC 13: test-retest consistency ────────────────────────────────────────
    # Within-person correlation between adjacent waves. Near-zero = mostly noise.
    print("\n[13] Test-retest: within-person correlation between adjacent waves (raw)")
    for dom, spec in DOMAIN_SPEC.items():
        wide = qc_df.pivot_table(index="participant_id", columns="session_id",
                                  values=spec["raw"], aggfunc="first")
        reported = False
        for w1, w2 in zip(DATA_WAVES, DATA_WAVES[1:]):
            if w1 not in wide.columns or w2 not in wide.columns:
                continue
            pair = wide[[w1, w2]].dropna()
            if len(pair) < 30:
                continue
            r = pair[w1].corr(pair[w2])
            print(f"    {dom:<28} {w1:<10} vs {w2:<10} n={len(pair):<6} r={r:+.3f}   "
                  f"[{_status(r > 0.20, warn=True)}]")
            reported = True
        if not reported:
            print(f"    {dom:<28} (only one wave with data -- test-retest not computable)")

    # ── QC 14: attrition / selection check ───────────────────────────────────
    # Compares baseline scores of returners vs non-returners.
    # Cognition-related dropout biases longitudinal estimates.
    print("\n[14] Attrition check: baseline scores of returners vs non-returners")
    if len(DATA_WAVES) >= 2:
        base_w, last_w = DATA_WAVES[0], DATA_WAVES[-1]
        returners = set(
            qc_df.loc[
                (qc_df["session_id"] == last_w)
                & qc_df[raw_cols_list].notna().any(axis=1),
                "participant_id",
            ]
        )
        base = qc_df[qc_df["session_id"] == base_w]
        for dom, spec in DOMAIN_SPEC.items():
            b = base[["participant_id", spec["raw"]]].dropna()
            if len(b) < 30:
                continue
            grp_ret = b.loc[b["participant_id"].isin(returners), spec["raw"]]
            grp_out = b.loc[~b["participant_id"].isin(returners), spec["raw"]]
            if len(grp_ret) < 10 or len(grp_out) < 10:
                continue
            n1, n2 = len(grp_ret), len(grp_out)
            sp = np.sqrt(((n1 - 1) * grp_ret.var(ddof=1) + (n2 - 1) * grp_out.var(ddof=1))
                         / (n1 + n2 - 2))
            d_val = (grp_ret.mean() - grp_out.mean()) / sp if sp > 0 else np.nan
            print(f"    {dom:<28} returners n={n1:<5} mean={grp_ret.mean():7.3f} | "
                  f"dropouts n={n2:<5} mean={grp_out.mean():7.3f} | "
                  f"Cohen's d={d_val:+.3f}   [{_status(abs(d_val) < 0.2, warn=True)}]")
        print(f"    (comparing {base_w} baseline scores; 'returners' = any cognitive "
              f"data at {last_w})")
    else:
        print("    (fewer than 2 waves with data -- attrition check not applicable)")

    # ── QC FIGURE 1: raw vs z-scored distributions ───────────────────────────
    print("\nBuilding QC figures …")
    n_dom_qc = len(DOMAIN_SPEC)
    figA, axesA = plt.subplots(n_dom_qc, 3, figsize=(15, 3.4 * n_dom_qc))
    figA.suptitle(
        "QC: raw vs z-scored distributions per domain\n"
        "Middle/right panels should be centred on 0 with SD 1 "
        "(dashed line = 0); shape is inherited from the raw score.",
        fontsize=12, y=1.005,
    )
    for i, (dom, spec) in enumerate(DOMAIN_SPEC.items()):
        zc_cohort = _zcol(dom, "_z_withincohort")
        zc_wave   = _zcol(dom, "_z_withincohortwave")
        panels = [
            (spec["raw"], f"{dom} — RAW\n({spec['units']})", "#4e79a7"),
            (zc_cohort,   f"{dom} — z within cohort",        "#f28e2b"),
            (zc_wave,     f"{dom} — z within cohort × wave", "#59a14f"),
        ]
        for j, (colname, title, colour) in enumerate(panels):
            ax = axesA[i, j]
            s = qc_df[colname].dropna()
            if s.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=9, fontweight="bold")
                continue
            ax.hist(s, bins=40, color=colour, alpha=0.8, edgecolor="white", linewidth=0.3)
            if j > 0:
                ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_ylabel("count", fontsize=8)
            ax.text(
                0.97, 0.95,
                f"n={len(s)}\nmean={s.mean():.2f}\nSD={s.std(ddof=1):.2f}",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    figA.tight_layout()
    outA = os.path.join(QC_DIR, "QC_distributions_ABCDv7.png")
    figA.savefig(outA, dpi=140, bbox_inches="tight")
    plt.close(figA)
    print(f"  Saved -> {outA}")

    # ── QC FIGURE 2: per-wave centring + raw→z monotonicity ──────────────────
    figB, axesB = plt.subplots(n_dom_qc, 3, figsize=(15, 3.4 * n_dom_qc))
    figB.suptitle(
        "QC: per-wave centring and raw→z monotonicity\n"
        "Col 1: z within cohort by wave (wave means differ — between-wave differences kept).\n"
        "Col 2: z within cohort×wave by wave (all wave means forced to 0).\n"
        "Col 3: raw vs z — must be a clean monotonic line in the expected direction.",
        fontsize=11, y=1.005,
    )
    for i, (dom, spec) in enumerate(DOMAIN_SPEC.items()):
        zc_cohort = _zcol(dom, "_z_withincohort")
        zc_wave   = _zcol(dom, "_z_withincohortwave")
        for j, (zc, title) in enumerate([
            (zc_cohort, f"{dom} — z within cohort, by wave"),
            (zc_wave,   f"{dom} — z within cohort × wave, by wave"),
        ]):
            ax = axesB[i, j]
            waves_plt, data_plt = [], []
            for wave, grp in qc_df.groupby("session_id"):
                s = grp[zc].dropna()
                if len(s) > 1:
                    waves_plt.append(wave)
                    data_plt.append(s.values)
            if not data_plt:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=9, fontweight="bold")
                continue
            ax.boxplot(data_plt, showfliers=False)
            ax.set_xticks(range(1, len(waves_plt) + 1))
            ax.set_xticklabels(waves_plt, rotation=45, fontsize=6)
            ax.axhline(0, color="#d62728", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_ylabel("z", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax = axesB[i, 2]
        pair = qc_df[[spec["raw"], zc_cohort]].dropna()
        if pair.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.scatter(pair[spec["raw"]], pair[zc_cohort], s=4, alpha=0.3,
                       color="#e15759", linewidths=0)
            expected_dir = "NEGATIVE (flipped)" if spec["flipped"] else "POSITIVE (not flipped)"
            ax.set_xlabel(f"raw ({spec['units']})", fontsize=8)
            ax.set_ylabel("z within cohort", fontsize=8)
            ax.axhline(0, color="#999999", linewidth=0.7)
            ax.text(0.03, 0.95, f"expect {expected_dir}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.85))
        ax.set_title(f"{dom} — raw vs z", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figB.tight_layout()
    outB = os.path.join(QC_DIR, "QC_centring_monotonicity_ABCDv7.png")
    figB.savefig(outB, dpi=140, bbox_inches="tight")
    plt.close(figB)
    print(f"  Saved -> {outB}")

    # ── QC FIGURE 3: cross-domain structure ───────────────────────────────────
    figC, axesC = plt.subplots(1, 3, figsize=(16, 4.8))
    figC.suptitle(
        "QC: cross-domain structure, floor/ceiling effects, and test-retest",
        fontsize=12, y=1.03,
    )

    # Panel 1: correlation heatmap
    ax = axesC[0]
    doms_qc = list(zcols_corr.keys())
    cmat = corr.values.astype(float)
    im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(doms_qc)))
    ax.set_yticks(range(len(doms_qc)))
    ax.set_xticklabels(doms_qc, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(doms_qc, fontsize=7)
    for ii in range(len(doms_qc)):
        for jj in range(len(doms_qc)):
            v = cmat[ii, jj]
            if not np.isnan(v):
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "#222222")
    ax.set_title("Cross-domain r (z within cohort,\nall higher = better)",
                 fontsize=9, fontweight="bold")
    figC.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: floor/ceiling bars
    ax = axesC[1]
    fc_doms, fc_lo_vals, fc_hi_vals = [], [], []
    for dom, spec in DOMAIN_SPEC.items():
        if spec["bounds"] is None:
            continue
        lo, hi = spec["bounds"]
        s = qc_df[spec["raw"]].dropna()
        if s.empty:
            continue
        fc_doms.append(dom)
        fc_lo_vals.append((s == lo).mean() * 100)
        fc_hi_vals.append((s == hi).mean() * 100)
    if fc_doms:
        x = np.arange(len(fc_doms))
        ax.bar(x - 0.2, fc_lo_vals, width=0.4, label="at floor", color="#4e79a7")
        ax.bar(x + 0.2, fc_hi_vals, width=0.4, label="at ceiling", color="#e15759")
        ax.axhline(FC_WARN, color="#888888", linestyle=":", linewidth=1,
                   label=f"{FC_WARN:.0f}% flag threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(fc_doms, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("% of observations", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("Floor / ceiling effects", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: test-retest correlations
    ax = axesC[2]
    palette_qc = {
        "IQ_scaled": "#4e79a7", "IQ_raw": "#9ecae1",
        "RespInhib_computed": "#f28e2b", "RespInhib_incongr_acc": "#fdae6b",
        "CogFlex": "#59a14f", "WorkingMem": "#e15759", "Attention": "#76b7b2",
    }
    tr_labels, tr_vals, tr_colors = [], [], []
    for dom, spec in DOMAIN_SPEC.items():
        wide = qc_df.pivot_table(index="participant_id", columns="session_id",
                                  values=spec["raw"], aggfunc="first")
        for w1, w2 in zip(DATA_WAVES, DATA_WAVES[1:]):
            if w1 not in wide.columns or w2 not in wide.columns:
                continue
            pair = wide[[w1, w2]].dropna()
            if len(pair) < 30:
                continue
            tr_labels.append(f"{dom}\n{w1}→{w2}")
            tr_vals.append(pair[w1].corr(pair[w2]))
            tr_colors.append(palette_qc.get(dom, "#999999"))
    if tr_vals:
        ax.bar(range(len(tr_vals)), tr_vals, color=tr_colors)
        ax.set_xticks(range(len(tr_vals)))
        ax.set_xticklabels(tr_labels, rotation=45, ha="right", fontsize=5.0)
        ax.axhline(0.2, color="#888888", linestyle=":", linewidth=1)
        ax.set_ylim(-0.1, 1)
        ax.set_ylabel("r between waves", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not computable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Test-retest (adjacent waves)", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figC.tight_layout()
    outC = os.path.join(QC_DIR, "QC_structure_ABCDv7.png")
    figC.savefig(outC, dpi=140, bbox_inches="tight")
    plt.close(figC)
    print(f"  Saved -> {outC}")

    # ── QC summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    if n_fail == 0 and n_warn == 0:
        print("QC SUMMARY: all checks passed.")
    else:
        print(f"QC SUMMARY: {n_fail} FAIL, {n_warn} WARN "
              f"-- review the flagged lines above.")
    print("=" * 78)
