#!/usr/bin/env python
# coding: utf-8

# harmonising_cognition_v3_ABCDv7.py
# Function: Harmonises ABCD data across 5 cognitive domains.
#           v2 adds across-cohort z-scoring alongside within-site z-scoring.
#           v3 changes the Millisecond Flanker z-score to use accuracy only
#              (dropping the v2 acc+RT composite). Rationale: the NIH Toolbox
#              Flanker handles speed-accuracy internally via its two-stage
#              algorithm; for consistency, and because the Millisecond task
#              has only 20 trials making median RT noisy, accuracy alone is
#              used as the Millisecond Flanker z-score input.
#              RespInhib_medrt_notz is retained in the output as a raw
#              reference variable but is not used in z-scoring.
#
# Z-scoring variants produced per domain
# ----------------------------------------
#   _z_withinsite      : z-scored within each ABCD site
#   _z_withinsitewave  : z-scored within each site × wave combination
#   _z_withincohort    : z-scored across all sites (full cohort as one group)
#   _z_withincohortwave: z-scored across all sites but separately within each wave
#
# Output: harmonised_cognition_v3_ABCDv7.csv


import argparse
import os
import pandas as pd
import numpy as np


# ── CLI arguments ──────────────────────────────────────────────────────────────
try:
    default_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    default_dir = os.getcwd()

parser = argparse.ArgumentParser(description="Harmonise ABCD cognitive data (v3).")
parser.add_argument(
    "--data-dir",
    default=default_dir,
    help="Directory containing the input TSV files.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Full path for the output CSV. "
         "Default: harmonised_cognition_v3_ABCDv7.csv inside --data-dir.",
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
    else os.path.join(DATA_DIR, "harmonised_cognition_v3_ABCDv7.csv")
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

# Within-site
iq["IQ_raw_z_withinsite"]    = zscore_within_groups(iq["IQ_raw_notz"],    [iq["site"]])
iq["IQ_scaled_z_withinsite"] = zscore_within_groups(iq["IQ_scaled_notz"], [iq["site"]])
# Within-site × wave
iq["IQ_raw_z_withinsitewave"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [iq["site"], iq["session_id"]])
iq["IQ_scaled_z_withinsitewave"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [iq["site"], iq["session_id"]])
# Across-cohort
iq["IQ_raw_z_withincohort"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [_cohort_key(iq.index)])
iq["IQ_scaled_z_withincohort"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [_cohort_key(iq.index)])
# Across-cohort × wave
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
nihtb_flnkr = nihtb[["participant_id", "session_id",
                       "nc_y_nihtb__flnkr_computed_score"]].copy()
nihtb_flnkr = nihtb_flnkr.rename(
    columns={"nc_y_nihtb__flnkr_computed_score": "nihtb_computed_score"})

# 2b. Millisecond Flanker — load acc and medrt; only acc is used for z-scoring
ms_flnkr = flnkr[["participant_id", "session_id",
                    "nc_y_flnkr_acc", "nc_y_flnkr_medrt"]].copy()
ms_flnkr = ms_flnkr.rename(columns={
    "nc_y_flnkr_acc":   "ms_acc",
    "nc_y_flnkr_medrt": "ms_medrt",
})

# 2c. Merge and assign task_type
# Rule: NIH Toolbox Flanker preferred; Millisecond used only when NIH unavailable.
ri = nihtb_flnkr.merge(ms_flnkr, on=["participant_id", "session_id"], how="outer")
ri = ri.merge(site_df,            on=["participant_id", "session_id"], how="left")

ri["task_type"] = pd.NA
ri.loc[ri["nihtb_computed_score"].notna(), "task_type"] = "nihtb_flanker"
ri.loc[
    ri["nihtb_computed_score"].isna() &
    (ri["ms_acc"].notna() | ri["ms_medrt"].notna()),
    "task_type"
] = "millisecond_flanker"

# 2d. Raw output columns
ri["RespInhib_computed_notz"] = np.where(
    ri["task_type"] == "nihtb_flanker", ri["nihtb_computed_score"], np.nan)
ri["RespInhib_acc_notz"] = np.where(
    ri["task_type"] == "millisecond_flanker", ri["ms_acc"], np.nan)

nihtb_mask = ri["task_type"] == "nihtb_flanker"
ms_mask    = ri["task_type"] == "millisecond_flanker"

# ── Z-scoring ─────────────────────────────────────────────────────────────────
# NIH Toolbox Flanker: higher computed_score = better → no flip (unchanged from v2)
# Millisecond Flanker: accuracy only, higher = better → no flip
#   (v2 used mean(acc_z, −RT_z); v3 drops RT from z-scoring entirely)

for suffix, site_groups, ms_groups in [
    ("_withinsite",       [ri["site"]],                              [ri["site"]]),
    ("_withinsitewave",   [ri["site"], ri["session_id"]],            [ri["site"], ri["session_id"]]),
    ("_withincohort",     [_cohort_key(ri.index)],                   [_cohort_key(ri.index)]),
    ("_withincohortwave", [ri["session_id"]],                        [ri["session_id"]]),
]:
    # NIH Toolbox Flanker z-score
    ri.loc[nihtb_mask, f"_nihtb_z{suffix}"] = zscore_within_groups(
        ri.loc[nihtb_mask, "nihtb_computed_score"],
        [g.loc[nihtb_mask] for g in site_groups],
    )
    # Millisecond Flanker z-score: accuracy only
    ri.loc[ms_mask, f"_ms_z{suffix}"] = zscore_within_groups(
        ri.loc[ms_mask, "ms_acc"],
        [g.loc[ms_mask] for g in ms_groups],
    )
    # Combine into final output column
    col = f"RespInhib_z{suffix}"
    ri[col] = np.nan
    ri.loc[nihtb_mask, col] = ri.loc[nihtb_mask, f"_nihtb_z{suffix}"]
    ri.loc[ms_mask,    col] = ri.loc[ms_mask,    f"_ms_z{suffix}"]

ri_out = ri[["participant_id", "session_id",
             "RespInhib_computed_notz", "RespInhib_acc_notz",
             "RespInhib_z_withinsite",       "RespInhib_z_withinsitewave",
             "RespInhib_z_withincohort",     "RespInhib_z_withincohortwave",
             "task_type"]]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 – Cognitive Flexibility  (NIH Toolbox DCCS)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Domain 3: Cognitive Flexibility …")

cf = nihtb[["participant_id", "session_id",
             "nc_y_nihtb__crdst_computed_score"]].copy()
cf = cf.rename(columns={"nc_y_nihtb__crdst_computed_score": "CogFlex_computed_notz"})
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
    "RespInhib_computed_notz", "RespInhib_acc_notz",
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
combined = combined.where(combined.notna(), other="n/a")


# ── Save ───────────────────────────────────────────────────────────────────────
combined.to_csv(OUT_FILE, index=False, na_rep="n/a")
print(f"\nDone.  Output saved to: {OUT_FILE}")
print(f"Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
print("\nColumn list:")
for c in combined.columns:
    print(f"  {c}")
