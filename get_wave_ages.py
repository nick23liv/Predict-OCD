#!/usr/bin/env python3
"""
get_wave_ages.py
================

Extracts the AGE AT EACH WAVE for one cohort and writes it as a small aggregate
table, so `plot_coverage_across_cohorts.py` can put real ages on its x-axis
instead of the planned ages from the project spreadsheet.

Run it once per cohort. It reads that cohort's own source data using exactly the
same age variable its harmonisation script uses, and writes:

    wave_ages_<COHORT>.csv
        cohort, session_id, n, age_mean, age_sd, age_median, age_p10, age_p90,
        age_min, age_max

NOTHING ELSE LEAVES THIS SCRIPT. No participant identifier, no individual age,
no cognitive score is printed or written -- only per-wave summary statistics.
Waves with fewer than --min-n participants are suppressed entirely (default 10),
so no cell can describe a handful of people.

USAGE
-----
    python3 get_wave_ages.py --cohort BHRC
    python3 get_wave_ages.py --cohort ABCD   --input /path/to/ab_g_dyn.tsv
    python3 get_wave_ages.py --cohort IMAGEN --input /rds/prj/cap_panda_ocd/imagen_2.7
    python3 get_wave_ages.py --cohort TEDS   --input "/path/to/810 EK FINAL 01102026.sav"

    # write straight into the FINAL_OUTPUTS tree the plot script reads
    python3 get_wave_ages.py --cohort BHRC --out-dir FINAL_OUTPUTS/BHRC/output

WHERE EACH AGE COMES FROM
-------------------------
Taken from each cohort's harmonisation script, so these are the same ages its QC
age-association checks already use:

  BHRC    BHRC_behaviour.csv            `age`  ("Age (years) at Domicile
                                        Evaluation"), long format, one row per
                                        ext_genid x redcap_event_name.

  ABCD    ab_g_dyn.tsv                  `ab_g_dyn__visit_age`, long format,
                                        already participant_id x session_id.

  IMAGEN  <root>/<WAVE>/meta_data/      MRI age at acquisition, recorded in DAYS.
          *age*acquisition*.csv         At BL/FU2 there is one column per scan
                                        sequence and a cell may hold two
                                        slash-separated days, so the median
                                        across sequences is taken and the first
                                        number in any such cell. The CANTAB `Age`
                                        column is deliberately NOT used: it is
                                        truncated to whole years and contains
                                        junk clustered at 0 and 9-11.

  TEDS    810 EK FINAL 01102026.sav     Per-wave age variables, resolved to each
                                        row's OWN twin value via the `twin`
                                        birth-order flag -- the 1/2 suffixed
                                        columns hold BOTH twins' values on every
                                        row, so taking "1" unconditionally would
                                        give every younger twin their sibling's
                                        age. Parent-report ages at 2/3/4/7/9y are
                                        family-level and carry no suffix.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)      # harmonizing_cognition/, one level up

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cohort", required=True,
                choices=["BHRC", "ABCD", "IMAGEN", "TEDS"],
                help="Which cohort to extract.")
ap.add_argument("--input", default=None,
                help="Source file (BHRC/ABCD/TEDS) or release root (IMAGEN). "
                     "Defaults to the path that cohort's harmonisation script uses.")
ap.add_argument("--out-dir", default=os.path.join(_HERE, "figures"),
                help="Where to write wave_ages_<COHORT>.csv (default: next to this script).")
ap.add_argument("--min-n", type=int, default=10,
                help="Suppress any wave with fewer than this many aged participants "
                     "(disclosure control). Default 10.")
args = ap.parse_args()

DAYS_PER_YEAR = 365.25


def finish(df: pd.DataFrame) -> None:
    """Summarise long-format participant-level ages into one row per wave."""
    df = df.dropna(subset=["age"])
    if df.empty:
        sys.exit(f"No ages found for {args.cohort}.")

    # Guard against a months-coded age variable slipping through.
    if df["age"].median() > 100:
        print("  [note] median age > 100 -- treating the variable as MONTHS and "
              "dividing by 12. Check this is right for your extract.")
        df["age"] = df["age"] / 12.0

    g = df.groupby("session_id")["age"]
    out = pd.DataFrame({
        "cohort":     args.cohort,
        "session_id": g.size().index,
        "n":          g.size().values,
        "age_mean":   g.mean().values,
        "age_sd":     g.std(ddof=1).values,
        "age_median": g.median().values,
        "age_p10":    g.quantile(0.10).values,
        "age_p90":    g.quantile(0.90).values,
        "age_min":    g.min().values,
        "age_max":    g.max().values,
    })

    small = out[out["n"] < args.min_n]
    for _, r in small.iterrows():
        print(f"  [suppressed] {r['session_id']}: n={int(r['n'])} < --min-n "
              f"{args.min_n}")
    out = out[out["n"] >= args.min_n].copy()
    if out.empty:
        sys.exit("Every wave was suppressed by --min-n; nothing to write.")

    for c in out.columns:
        if c.startswith("age_"):
            out[c] = out[c].round(2)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"wave_ages_{args.cohort}.csv")
    out.to_csv(path, index=False)

    print(f"\nAges per wave for {args.cohort}:")
    print(out.to_string(index=False))
    print(f"\nSaved -> {path}")
    print("Aggregate statistics only -- safe to share.")


# ── BHRC ───────────────────────────────────────────────────────────────────────
def bhrc():
    path = args.input or os.path.join(_ROOT, "harmonising", "BHRC",
                                      "BHRC_behaviour.csv")
    print(f"Reading {path}")
    # usecols keeps this to 3 columns out of ~5,000.
    df = pd.read_csv(path, usecols=["ext_genid", "redcap_event_name", "age"],
                     na_values="n/a", low_memory=False)
    df = df.rename(columns={"ext_genid": "participant_id",
                            "redcap_event_name": "session_id"})
    return finish(df[["participant_id", "session_id", "age"]])


# ── ABCD ───────────────────────────────────────────────────────────────────────
def abcd():
    path = args.input or os.path.join(_ROOT, "harmonising", "toydata_25june2026",
                                      "ab_g_dyn.tsv")
    print(f"Reading {path}")
    df = pd.read_csv(path, sep="\t", na_values="n/a", low_memory=False,
                     usecols=["participant_id", "session_id", "ab_g_dyn__visit_age"])
    df = df.rename(columns={"ab_g_dyn__visit_age": "age"})
    df = df.drop_duplicates(subset=["participant_id", "session_id"], keep="first")
    return finish(df)


# ── IMAGEN ─────────────────────────────────────────────────────────────────────
# Mirrors load_mri_age() in harmonising_cognition_IMAGEN.py, including the
# per-wave plausible bands that script applies before using the values.
IMAGEN_WAVES = ["BL", "FU2", "FU3"]
IMAGEN_AGE_BANDS = {"BL": (12, 17), "FU2": (16, 24), "FU3": (19, 27)}


def imagen():
    root = args.input or "/rds/prj/cap_panda_ocd/imagen_2.7"
    print(f"Reading IMAGEN release at {root}")
    frames = []
    for wave in IMAGEN_WAVES:
        hits = sorted(glob.glob(os.path.join(root, wave, "meta_data",
                                             "*age*acquisition*.csv")))
        if len(hits) != 1:
            print(f"  [warn] {wave}: expected 1 MRI-age file, found {len(hits)} "
                  f"-- skipping this wave")
            continue
        raw = pd.read_csv(hits[0], dtype=str)

        def _first_number(x):
            if pd.isna(x):
                return np.nan
            found = re.findall(r"\d+(?:\.\d+)?", str(x))
            return float(found[0]) if found else np.nan

        values = raw.drop(columns=["PSC2"]).apply(lambda c: c.map(_first_number))
        f = pd.DataFrame({
            "participant_id": raw["PSC2"],
            "session_id": wave,
            "age": values.median(axis=1, skipna=True) / DAYS_PER_YEAR,
        })
        lo, hi = IMAGEN_AGE_BANDS[wave]
        bad = (f["age"] < lo) | (f["age"] > hi)
        f.loc[bad, "age"] = np.nan
        print(f"  {wave:<4} {len(f):>5} rows, blanked {int(bad.sum()):>3} outside {lo}-{hi}")
        frames.append(f)
    if not frames:
        sys.exit("No IMAGEN MRI-age files found.")
    return finish(pd.concat(frames, ignore_index=True))


# ── TEDS ───────────────────────────────────────────────────────────────────────
# (session_id, age variable base). Bases with "1"/"2" suffixed columns are
# resolved per row via the twin flag; the parent-report ages (bpbage..icpage) are
# family-level and used as-is. Taken from the wave tables in
# harmonising_cognition_TEDS.py.
TEDS_WAVE_AGES = [
    ("age02",    "bpbage"),
    ("age03",    "cpbage"),
    ("age04",    "dpbage"),
    ("age07",    "gpbage"),
    ("age09",    "icpage"),
    ("age10",    "jcstage"),
    ("age12",    "lcqage"),
    ("age14",    "ncqage"),
    ("age16",    "pcbhage"),
    ("teds21",   "u1cage"),
    ("catslife", "zcage"),
]


def teds():
    path = args.input or os.path.join(_ROOT, "harmonising", "TEDS",
                                      "810 EK FINAL 01102026.sav")
    print(f"Reading {path}")

    wanted = ["id_twin", "twin"]
    for _, base in TEDS_WAVE_AGES:
        wanted += [base, f"{base}1", f"{base}2"]

    # Column names must be discovered first, because each wave's age may be
    # stored suffixed ("pcbhage1"/"pcbhage2") or unsuffixed (the family-level
    # parent-report ages). Read the METADATA ONLY -- an earlier version called
    # pd.read_spss() with no usecols for this, which pulls all ~7,750 columns
    # into memory and is killed by the OOM reaper on a login node.
    try:
        import pyreadstat
        _, meta = pyreadstat.read_sav(path, metadataonly=True)
        allcols = set(meta.column_names)
    except ImportError:
        sys.exit("TEDS needs pyreadstat (pandas uses it for .sav files):\n"
                 "    pip install pyreadstat")

    use = [c for c in dict.fromkeys(wanted) if c in allcols]
    missing = [c for c in dict.fromkeys(wanted) if c not in allcols]
    if not use:
        sys.exit("None of the expected TEDS age columns were found.")
    print(f"  {len(allcols)} columns in file; reading {len(use)} of them")
    raw = pd.read_spss(path, convert_categoricals=False, usecols=use)
    print(f"  {len(raw)} rows loaded")

    frames = []
    for session_id, base in TEDS_WAVE_AGES:
        c1, c2 = f"{base}1", f"{base}2"
        if c1 in raw.columns and c2 in raw.columns:
            # Suffixed: both twins' values sit on every row -- pick this row's own.
            age = np.where(raw["twin"] == 1, raw[c1], raw[c2])
        elif base in raw.columns:
            age = raw[base]              # family-level, one shared value
        else:
            print(f"  [warn] {session_id}: no age column ({base} / {c1} / {c2}) "
                  f"-- skipping")
            continue
        frames.append(pd.DataFrame({"participant_id": raw["id_twin"],
                                    "session_id": session_id,
                                    "age": age}))
    if not frames:
        sys.exit("No TEDS age columns found.")
    return finish(pd.concat(frames, ignore_index=True))


{"BHRC": bhrc, "ABCD": abcd, "IMAGEN": imagen, "TEDS": teds}[args.cohort]()
