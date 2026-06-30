#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: harmonise_cognition_ABCDv7.ipynb
# Function: Harmonises ABCD data across 5 cognitive domains:
#           (IQ, Response Inhibition, Cognitive Flexibility, Working Memory, Attention)
# Output: harmonised_cognition_ABCDv7.csv


# In[2]:


import argparse
import os
import pandas as pd
import numpy as np


# In[20]:


# CLI arguments 
try:
    default_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running in Jupyter
    default_dir = os.getcwd()

parser = argparse.ArgumentParser(description="Harmonise ABCD cognitive data.")
parser.add_argument(
    "--data-dir",
    default=default_dir,
    help="Directory containing the input TSV files "
         "(default: same directory as this script).",
)
parser.add_argument(
    "--output",
    default=None,
    help="Full path for the output CSV. "
         "Default: harmonised_cognition.csv inside --data-dir.",
)

# Parse arguments
try:
    __file__
    # Running as a script
    args = parser.parse_args()
except NameError:
    # Running in Jupyter
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
    else os.path.join(DATA_DIR, "harmonised_cognition_ABCDv7.csv")
)


# In[9]:


# Utility: robust within-group z-score 
def zscore_within_groups(series: pd.Series, groups: pd.Series) -> pd.Series:
    def _safe_z(x):
        n   = x.notna().sum()
        std = x.std(ddof=1)
        if n <= 1 or std == 0 or pd.isna(std):
            # Can't compute a meaningful z-score; assign 0 (group mean).
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


# In[10]:


# Load files
print("Loading data files …")
dyn   = pd.read_csv(FILES["dyn"],   sep="\t", na_values="n/a", low_memory=False)
wisc  = pd.read_csv(FILES["wisc"],  sep="\t", na_values="n/a", low_memory=False)
flnkr = pd.read_csv(FILES["flnkr"], sep="\t", na_values="n/a", low_memory=False)
nihtb = pd.read_csv(FILES["nihtb"], sep="\t", na_values="n/a", low_memory=False)
lmt   = pd.read_csv(FILES["lmt"],   sep="\t", na_values="n/a", low_memory=False)


# In[11]:


# Extract site (one row per participant × session) 
print("Extracting site information …")
site_df = (
    dyn[["participant_id", "session_id", "ab_g_dyn__design_site"]]
    .dropna(subset=["ab_g_dyn__design_site"])
    .drop_duplicates(subset=["participant_id", "session_id"], keep="first")
    .rename(columns={"ab_g_dyn__design_site": "site"})
)


# In[12]:


# Build the master spine (union of all participant × session rows) 
print("Building master participant × session spine …")
all_ids = pd.concat([
    wisc [["participant_id", "session_id"]],
    flnkr[["participant_id", "session_id"]],
    nihtb[["participant_id", "session_id"]],
    lmt  [["participant_id", "session_id"]],
    site_df[["participant_id", "session_id"]],
]).drop_duplicates()

master = all_ids.merge(site_df, on=["participant_id", "session_id"], how="left")


# In[13]:


### DOMAIN 1 – IQ  (WISC-V Matrix Reasoning) ###

print("Processing Domain 1: IQ …")

iq = wisc[["participant_id", "session_id",
           "nc_y_wisc__raw_score", "nc_y_wisc__scaled_score"]].copy()

# Rename to output labels (raw, unharmonised values)
iq = iq.rename(columns={
    "nc_y_wisc__raw_score":    "IQ_raw_notz",
    "nc_y_wisc__scaled_score": "IQ_scaled_notz",
})

# Merge site so we can compute within-site z-scores
iq = iq.merge(site_df, on=["participant_id", "session_id"], how="left")

# Within-site z-scores
iq["IQ_raw_z_withinsite"]    = zscore_within_groups(iq["IQ_raw_notz"],    [iq["site"]])
iq["IQ_scaled_z_withinsite"] = zscore_within_groups(iq["IQ_scaled_notz"], [iq["site"]])

# Within-site × wave z-scores
iq["IQ_raw_z_withinsitewave"]    = zscore_within_groups(
    iq["IQ_raw_notz"],    [iq["site"], iq["session_id"]])
iq["IQ_scaled_z_withinsitewave"] = zscore_within_groups(
    iq["IQ_scaled_notz"], [iq["site"], iq["session_id"]])

iq_out = iq[["participant_id", "session_id",
             "IQ_raw_notz", "IQ_scaled_notz",
             "IQ_raw_z_withinsite", "IQ_scaled_z_withinsite",
             "IQ_raw_z_withinsitewave", "IQ_scaled_z_withinsitewave"]]


# In[15]:


### DOMAIN 2 – Response Inhibition ###

print("Processing Domain 2: Response Inhibition …")

# 2a. Get NIH Toolbox Flanker scores 
nihtb_flnkr = nihtb[["participant_id", "session_id",
                       "nc_y_nihtb__flnkr_computed_score"]].copy()
nihtb_flnkr = nihtb_flnkr.rename(
    columns={"nc_y_nihtb__flnkr_computed_score": "nihtb_computed_score"})

# 2b. Get Millisecond Flanker scores 
ms_flnkr = flnkr[["participant_id", "session_id",
                    "nc_y_flnkr_acc", "nc_y_flnkr_medrt"]].copy()
ms_flnkr = ms_flnkr.rename(columns={
    "nc_y_flnkr_acc":   "ms_acc",
    "nc_y_flnkr_medrt": "ms_medrt",
})

# 2c. Merge both task sources 
ri = nihtb_flnkr.merge(ms_flnkr, on=["participant_id", "session_id"], how="outer")
ri = ri.merge(site_df,            on=["participant_id", "session_id"], how="left")

# 2d. Assign task_type and select scores
# Rule: if NIH Toolbox Flanker is available, use it; otherwise use Millisecond.
# "Available" = computed_score is not NaN.
ri["task_type"] = pd.NA

# Prefer NIH Toolbox Flanker if available
ri.loc[
    ri["nihtb_computed_score"].notna(),
    "task_type"
] = "nihtb_flanker"

# Otherwise use Millisecond Flanker
ri.loc[
    ri["nihtb_computed_score"].isna() &
    (ri["ms_acc"].notna() | ri["ms_medrt"].notna()),
    "task_type"
] = "millisecond_flanker"

# Raw (unharmonised) output columns
ri["RespInhib_computed_notz"] = np.where(
    ri["task_type"] == "nihtb_flanker", ri["nihtb_computed_score"], np.nan)
ri["RespInhib_acc_notz"]   = np.where(
    ri["task_type"] == "millisecond_flanker", ri["ms_acc"],   np.nan)
ri["RespInhib_medrt_notz"] = np.where(
    ri["task_type"] == "millisecond_flanker", ri["ms_medrt"], np.nan)

# 2e. Within-site z-scores, computed separately per task type
# We z-score each task within (site, task_type) so the scales are comparable before we place them in a single column.
# NIH Toolbox Flanker: higher computed_score = better  → no flip
nihtb_mask = ri["task_type"] == "nihtb_flanker"
ri.loc[nihtb_mask, "_nihtb_z_withinsite"] = zscore_within_groups(
    ri.loc[nihtb_mask, "nihtb_computed_score"],
    [ri.loc[nihtb_mask, "site"]]
)
ri.loc[nihtb_mask, "_nihtb_z_withinsitewave"] = zscore_within_groups(
    ri.loc[nihtb_mask, "nihtb_computed_score"],
    [ri.loc[nihtb_mask, "site"], ri.loc[nihtb_mask, "session_id"]]
)

# Millisecond Flanker: z-score acc and rt separately, then combine.
#  Accuracy:  higher = better  → no flip
#  Median RT: higher = slower = worse  → multiply z by –1
ms_mask = ri["task_type"] == "millisecond_flanker"

ri.loc[ms_mask, "_ms_acc_z_withinsite"] = zscore_within_groups(
    ri.loc[ms_mask, "ms_acc"],
    [ri.loc[ms_mask, "site"]]
)
ri.loc[ms_mask, "_ms_rt_z_withinsite"] = zscore_within_groups(
    ri.loc[ms_mask, "ms_medrt"],
    [ri.loc[ms_mask, "site"]]
)
# Flip RT z-score so higher = faster = better, then average with acc z
ri.loc[ms_mask, "_ms_z_withinsite"] = (
    ri.loc[ms_mask, "_ms_acc_z_withinsite"]
    + (-1 * ri.loc[ms_mask, "_ms_rt_z_withinsite"])
) / 2.0

ri.loc[ms_mask, "_ms_acc_z_withinsitewave"] = zscore_within_groups(
    ri.loc[ms_mask, "ms_acc"],
    [ri.loc[ms_mask, "site"], ri.loc[ms_mask, "session_id"]]
)
ri.loc[ms_mask, "_ms_rt_z_withinsitewave"] = zscore_within_groups(
    ri.loc[ms_mask, "ms_medrt"],
    [ri.loc[ms_mask, "site"], ri.loc[ms_mask, "session_id"]]
)
ri.loc[ms_mask, "_ms_z_withinsitewave"] = (
    ri.loc[ms_mask, "_ms_acc_z_withinsitewave"]
    + (-1 * ri.loc[ms_mask, "_ms_rt_z_withinsitewave"])
) / 2.0

# 2f. Combine into final response inhibition z-score columns 
ri["RespInhib_z_withinsite"] = np.nan

ri.loc[
    ri["task_type"] == "nihtb_flanker",
    "RespInhib_z_withinsite"
] = ri.loc[
    ri["task_type"] == "nihtb_flanker",
    "_nihtb_z_withinsite"
]

ri.loc[
    ri["task_type"] == "millisecond_flanker",
    "RespInhib_z_withinsite"
] = ri.loc[
    ri["task_type"] == "millisecond_flanker",
    "_ms_z_withinsite"
]

ri["RespInhib_z_withinsitewave"] = np.nan

ri.loc[
    ri["task_type"] == "nihtb_flanker",
    "RespInhib_z_withinsitewave"
] = ri.loc[
    ri["task_type"] == "nihtb_flanker",
    "_nihtb_z_withinsitewave"
]

ri.loc[
    ri["task_type"] == "millisecond_flanker",
    "RespInhib_z_withinsitewave"
] = ri.loc[
    ri["task_type"] == "millisecond_flanker",
    "_ms_z_withinsitewave"
]

ri_out = ri[["participant_id", "session_id",
             "RespInhib_computed_notz",
             "RespInhib_acc_notz",
             "RespInhib_medrt_notz",
             "RespInhib_z_withinsite",
             "RespInhib_z_withinsitewave",
             "task_type"]]


# In[16]:


### DOMAIN 3 – Cognitive Flexibility  (NIH Toolbox DCCS) ###

print("Processing Domain 3: Cognitive Flexibility …")

cf = nihtb[["participant_id", "session_id",
             "nc_y_nihtb__crdst_computed_score"]].copy()
cf = cf.rename(columns={"nc_y_nihtb__crdst_computed_score": "CogFlex_computed_notz"})
cf = cf.merge(site_df, on=["participant_id", "session_id"], how="left")

# Higher computed score = better flexible switching  → no flip
cf["CogFlex_z_withinsite"]     = zscore_within_groups(
    cf["CogFlex_computed_notz"], [cf["site"]])
cf["CogFlex_z_withinsitewave"] = zscore_within_groups(
    cf["CogFlex_computed_notz"], [cf["site"], cf["session_id"]])

cf_out = cf[["participant_id", "session_id",
             "CogFlex_computed_notz",
             "CogFlex_z_withinsite",
             "CogFlex_z_withinsitewave"]]


# In[17]:


### DOMAIN 4 – Working Memory  (NIH Toolbox List Sorting) ###

print("Processing Domain 4: Working Memory …")

wm = nihtb[["participant_id", "session_id",
             "nc_y_nihtb__lswmt__raw_score"]].copy()
wm = wm.rename(columns={"nc_y_nihtb__lswmt__raw_score": "WorkingMem_raw_notz"})
wm = wm.merge(site_df, on=["participant_id", "session_id"], how="left")

# Higher raw score = more correct reorderings = better  → no flip
wm["WorkingMem_z_withinsite"]     = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [wm["site"]])
wm["WorkingMem_z_withinsitewave"] = zscore_within_groups(
    wm["WorkingMem_raw_notz"], [wm["site"], wm["session_id"]])

wm_out = wm[["participant_id", "session_id",
             "WorkingMem_raw_notz",
             "WorkingMem_z_withinsite",
             "WorkingMem_z_withinsitewave"]]


# In[18]:


### DOMAIN 5 – Attention  (Little Man Task) ###

print("Processing Domain 5: Attention …")

att = lmt[["participant_id", "session_id",
            "nc_y_lmt__crct_acc"]].copy()
att = att.rename(columns={"nc_y_lmt__crct_acc": "Attention_raw_notz"})
att = att.merge(site_df, on=["participant_id", "session_id"], how="left")

# Higher accuracy = better attentional performance  → no flip
att["Attention_z_withinsite"]     = zscore_within_groups(
    att["Attention_raw_notz"], [att["site"]])
att["Attention_z_withinsitewave"] = zscore_within_groups(
    att["Attention_raw_notz"], [att["site"], att["session_id"]])

att_out = att[["participant_id", "session_id",
               "Attention_raw_notz",
               "Attention_z_withinsite",
               "Attention_z_withinsitewave"]]


# In[19]:


# COMBINE ALL DOMAINS

print("Combining all domains …")

combined = master.copy()
for df in [iq_out, ri_out, cf_out, wm_out, att_out]:
    combined = combined.merge(df, on=["participant_id", "session_id"], how="left")

# Ensure task_type is preserved as a string label (not NaN for missing rows)
combined["task_type"] = combined["task_type"].fillna("n/a")

# Column ordering
col_order = [
    "participant_id", "session_id", "site",
    # IQ
    "IQ_raw_notz", "IQ_scaled_notz",
    "IQ_raw_z_withinsite", "IQ_scaled_z_withinsite",
    "IQ_raw_z_withinsitewave", "IQ_scaled_z_withinsitewave",
    # Response Inhibition
    "RespInhib_computed_notz", "RespInhib_acc_notz", "RespInhib_medrt_notz",
    "RespInhib_z_withinsite", "RespInhib_z_withinsitewave",
    "task_type",
    # Cognitive Flexibility
    "CogFlex_computed_notz", "CogFlex_z_withinsite", "CogFlex_z_withinsitewave",
    # Working Memory
    "WorkingMem_raw_notz", "WorkingMem_z_withinsite", "WorkingMem_z_withinsitewave",
    # Attention
    "Attention_raw_notz", "Attention_z_withinsite", "Attention_z_withinsitewave",
]
combined = combined[col_order]

# Replace NaN with "n/a" for consistent missing-data representation
combined = combined.where(combined.notna(), other="n/a")


# In[21]:


# SAVE

combined.to_csv(OUT_FILE, index=False, na_rep="n/a")
print(f"\nDone.  Output saved to: {OUT_FILE}")
print(f"Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
print("\nColumn list:")
for c in combined.columns:
    print(f"  {c}")

