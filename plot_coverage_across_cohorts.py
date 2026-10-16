#!/usr/bin/env python3
"""
plot_coverage_across_cohorts.py
===============================

Cross-cohort coverage figure: how many participants ACTUALLY have usable data
for each cognitive domain, at each wave, in each cohort.

Every number here is counted from the real harmonised output CSVs:

    FINAL_OUTPUTS/<COHORT>/output/harmonised_cognition_<COHORT>.csv

This is the observed-data counterpart to `plotting_predict_cognition.R`, which
plots the *planned* measures and ages from `predict_ocd_sites_cognition_*.csv`.
The palette, cohort order and domain order are copied
from that script so the two figures sit together in a deck; the numbers are not
-- nothing is taken from the planning spreadsheets.

Only counts of non-missing values are read, aggregated and written. No
individual-level value is ever printed, plotted or saved.

DOMAIN DETECTION
----------------
Domains are matched by column name rather than hard-coded, because the middle
token differs between cohorts (IQ_raw_notz in BHRC, IQ_scaled_notz in ABCD):

    ^(IQ|RespInhib|CogFlex|WorkingMem|Attention)_.*_notz$

If a cohort has MORE THAN ONE column for the same domain, the one with the
largest non-missing N is used and the choice is printed. A cohort with no column
for a domain contributes nothing to that panel.

Where a cohort switched instrument between waves within a domain -- ABCD's
flanker variants being the example -- that is treated as one continuous measure
for the domain, which is what the upstream harmonisation already encodes by
writing a single column.

USAGE
-----
    # real data -- auto-discovers every cohort present
    python3 plot_coverage_across_cohorts.py

    # preview the layout on synthetic data (never touches FINAL_OUTPUTS)
    python3 plot_coverage_across_cohorts.py --synthetic

    # grouped bars instead of lines; log axis for the ABCD/BHRC size gap
    python3 plot_coverage_across_cohorts.py --style bar --log

OUTPUT
------
    coverage_across_cohorts[_bar][_log].png
    coverage_across_cohorts.csv    the counts behind the figure (cohort, domain,
                                   wave, n) -- aggregate only, safe to share
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

# matplotlib is needed only to DRAW. --counts-only runs inside a secure
# environment where the plotting stack may not be installed at all, and the
# counting half needs nothing but pandas, so a missing matplotlib is not fatal
# until a figure is actually requested.
try:
    import matplotlib
    matplotlib.use("Agg")                  # headless: never needs a display
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    _HAVE_MPL = True
except ImportError as _mpl_err:            # noqa: F841
    plt = mticker = Line2D = None
    _HAVE_MPL = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)      # harmonizing_cognition/, one level up

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--final-outputs", default=os.path.join(_ROOT, "FINAL_OUTPUTS"),
                help="Folder containing one subfolder per cohort.")
ap.add_argument("--out-dir", default=None,
                help="Where to write the figure + counts CSV (default: next to this script).")
ap.add_argument("--style", choices=["line", "bar"], default="line",
                help="line = one trace per cohort across waves (default); "
                     "bar = grouped bars per wave.")
ap.add_argument("--layout", choices=["single", "facet"], default="single",
                help="single = every domain on ONE axes so counts are directly "
                     "comparable across domains (default); facet = one panel "
                     "per domain.")
ap.add_argument("--x", choices=["age", "wave"], default="age",
                help="x-axis: age in years (default, comparable across cohorts) "
                     "or wave index.")
ap.add_argument("--figure", choices=["lines", "stacked", "both"], default="lines",
                help="lines = per-cohort trajectories (default); stacked = one "
                     "panel per domain, bars showing TOTAL participants "
                     "available at each age with cohorts stacked; both = write "
                     "each to its own PNG.")
ap.add_argument("--bin-width", type=float, default=1.0,
                help="Age bin width in years for the stacked figure. Cohorts "
                     "are assessed at scattered ages, so counts must be binned "
                     "before they can be summed. Default 1.0.")
ap.add_argument("--ybreak", default="none",
                help="Break the count axis so small cohorts are not squashed by "
                     "large ones. 'none' (default) = continuous; 'auto' = split "
                     "at the widest empty band in the data; or give it "
                     "explicitly as LOW,HIGH (e.g. 4000,10000) to show 0-LOW "
                     "below the break and HIGH-upwards above it. Preferred over "
                     "--log, which compresses the differences you want to read.")
ap.add_argument("--log", action="store_true",
                help="Log-scale the count axis. Cohort sizes span ~2,200 "
                     "(Imagen) to ~16,800 (TEDS), so this can help.")
ap.add_argument("--annotate", action="store_true",
                help="Print the count beside every marker.")
ap.add_argument("--jitter", type=float, default=0.14,
                help="Horizontal offset between domains at the same x, in years "
                     "(single-panel layout only). Domains measured at the same "
                     "wave would otherwise sit on identical x and hide each "
                     "other's colour. 0 disables. Default 0.14.")
ap.add_argument("--keep-empty-waves", action="store_true",
                help="Keep waves that are empty for every domain (e.g. BHRC's "
                     "covid sub-waves). Dropped by default.")
ap.add_argument("--synthetic", action="store_true",
                help="Generate fake cohorts and plot those, for tuning the "
                     "figure's appearance. Writes nothing to FINAL_OUTPUTS.")
ap.add_argument("--input", nargs="+", default=[], metavar="[COHORT=]CSV",
                help="Harmonised CSV(s) to count, given directly instead of "
                     "discovered under --final-outputs. For an environment "
                     "where the file is not in <COHORT>/output/ and is not "
                     "named harmonised_cognition_<COHORT>.csv -- e.g. the ABCD "
                     "TRE. The cohort is read from the filename where it can "
                     "be ("
                     "harmonised_cognition_v4_ABCDv7.csv -> ABCD); prefix "
                     "'ABCD=' to say so explicitly.")
ap.add_argument("--counts-only", action="store_true",
                help="Count and write coverage_across_cohorts.csv, then stop "
                     "without drawing anything. For running INSIDE a secure "
                     "environment (e.g. the ABCD TRE) where the harmonised CSV "
                     "may not leave: the counts file it writes is aggregate "
                     "only and is what you export.")
ap.add_argument("--merge-counts", nargs="+", default=[], metavar="CSV",
                help="Counts CSVs produced elsewhere by --counts-only, merged "
                     "with whatever is discovered locally. This is how a cohort "
                     "whose individual data cannot leave its environment still "
                     "appears in the figures. A cohort present both locally and "
                     "in a merged file is taken from the merged file.")
ap.add_argument("--allow-no-local", action="store_true",
                help="Permit a run with no harmonised CSVs found locally, when "
                     "every cohort is arriving via --merge-counts.")
args = ap.parse_args()

OUT_DIR = args.out_dir if args.out_dir else os.path.join(_HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Domains ────────────────────────────────────────────────────────────────────
# Panel order top -> bottom, matching the planned-measures figure.
DOMAINS = ["Attention", "CogFlex", "RespInhib", "WorkingMem", "IQ"]

DOMAIN_LABEL = {
    "Attention":  "Attention",
    "CogFlex":    "Cognitive flexibility",
    "RespInhib":  "Response inhibition",
    "WorkingMem": "Working memory",
    "IQ":         "IQ",
}

RAW_COL_RE = re.compile(r"^(" + "|".join(DOMAINS) + r")_.*_notz$")

# Encoding: COLOUR carries the cognitive domain, MARKER SHAPE carries the
# cohort, and every line is solid.
#
# Five maximally separated hues -- blue / orange / green / red / purple. The
# earlier Paul Tol "bright" set put Attention on a magenta-purple and Working
# Memory on a salmon red, which read as near-identical at line width. These are
# assigned so the closest remaining pair (orange and red) falls on Working
# Memory and Cognitive Flexibility rather than on two domains that appear
# together in every cohort -- Cognitive Flexibility is absent from several
# cohorts entirely, so the two rarely compete in one panel.
DOMAIN_COLOURS = {
    "Attention":  "#9467BD",   # purple
    "CogFlex":    "#D62728",   # red
    "RespInhib":  "#2CA02C",   # green
    "WorkingMem": "#FF7F0E",   # orange
    "IQ":         "#1F77B4",   # blue
}

# Up to nine visually distinct shapes, assigned in COHORT_ORDER.
COHORT_MARKERS = {
    "ABCD": "o", "ALSPAC": "s", "Imagen": "^", "TEDS": "D",
    "TRAILS": "v", "CATSS": "P", "BHRC": "X", "cVEDA": "*", "Gen R": "h",
}
FALLBACK_MARKERS = ["<", ">", "p", "8", "H"]

# Line and marker weight. Every cohort in this project is a discovery cohort, so
# none is drawn heavier than another -- one line width, one marker size. Raise
# both together to scale the whole figure up.
LINE_WIDTH        = 4.2
MARKER_SIZE       = 14.0

# matplotlib sizes markers by bounding box, not ink area, so a diamond or square
# reads noticeably heavier than a triangle or cross at the same `markersize`,
# and a star reads lighter. These multipliers even that out by eye.
MARKER_SHAPE_SCALE = {
    "o": 1.00, "s": 0.90, "D": 0.82, "^": 1.06, "v": 1.06,
    "P": 1.04, "X": 1.04, "*": 1.30, "h": 0.96,
    "<": 1.06, ">": 1.06, "p": 1.00, "8": 0.96, "H": 0.96,
}

# ── Age on the x-axis ──────────────────────────────────────────────────────────
# Preferred source: a real age column in the cohort's harmonised CSV, from which
# the MEDIAN age per wave is taken (an aggregate -- no individual value is read
# out). None of the harmonised outputs carry age today, so the fallback below is
# what will actually be used until an age column is added to them.
#
# FALLBACK: nominal assessment ages, seeded from predict_ocd_sites_cognition_ages.csv
# (the PLANNED ages, not observed). Wave index -> age, by position. These are
# approximate and worth correcting per cohort as the real outputs land; every
# cohort plotted from this table is listed in a warning and in the figure footer.
AGE_COL_CANDIDATES = ["age", "age_years", "visit_age", "interview_age",
                      "ab_g_dyn__visit_age"]

# BEST source: a wave_ages_<COHORT>.csv produced by get_wave_ages.py, which pulls
# the real age from each cohort's own source data and summarises it per wave.
# Searched for next to the harmonised CSV and in this script's folder. These are
# aggregate statistics, so reading them discloses nothing.
AGE_TABLE_GLOBS = ["wave_ages_*.csv"]
AGE_TABLE_COL = "age_median"      # median is robust to the odd mis-coded age

# Cohort x domain wave exclusions by AGE, applied once ages are resolved.
#
# These figures show ALL available waves -- they are a picture of what data
# exists, not of what a particular model would use, so a wave the documentation
# excludes for analytic reasons still belongs here. The sole exception is
# TEDS's IQ composite at ages 2/3/4, which is parent-reported infant cognition
# rather than the same construct at all (its correlation with later IQ is
# r ~ 0.09-0.19, against ~0.63 from age16). Plotting those as "IQ coverage"
# would overstate what TEDS can contribute, so the cut stays at age 7.
WAVE_FILTERS = {
    ("TEDS", "IQ"): {"min_age": 7},
}

# A value may be either:
#   list -- ages by WAVE INDEX (position 0 = wave 0, 1 = wave 1, ...)
#   dict -- ages keyed by the cohort's own session_id, for cohorts whose waves
#           are not a contiguous 0,1,2,... run. IMAGEN is the case in point: its
#           waves are BL / FU2 / FU3, i.e. indices 0, 2, 3, because FU1 carries
#           no cognitive data. A positional list silently put FU2 at 22 and gave
#           FU3 nothing at all.
NOMINAL_AGES = {
    # Keyed, not positional, for the same reason as TEDS below: ABCD's session
    # labels are ses-00A / ses-02A / ses-04A / ses-06A, i.e. wave indices
    # 0, 2, 4, 6 -- the number is YEARS SINCE BASELINE, not a wave position.
    # A positional list would read ses-04A as the third wave and place it at 13
    # instead of ~13.9, and would run off the end for later follow-ups.
    # These are the planned ages; a wave_ages_ABCD.csv from get_wave_ages.py
    # overrides them with observed medians.
    "ABCD":   {"ses-00A": 9.9, "ses-01A": 10.9, "ses-02A": 11.9,
               "ses-03A": 12.9, "ses-04A": 13.9, "ses-05A": 14.9,
               "ses-06A": 15.9, "ses-07A": 16.9, "ses-08A": 17.9},
    "ALSPAC": [8.5, 10, 11.5, 15.5, 17.5, 24],
    "Imagen": {"BL": 14, "FU2": 19, "FU3": 22},
    # Keyed, not positional: TEDS session labels embed the age as a number, so
    # wave_index("age16") is 16, not a position in a list. A positional list put
    # age02/03/04 at ages 14/16/21 and dropped everything from age07 on --
    # including the only Working Memory and Attention waves, both age16.
    "TEDS":   {"age02": 2, "age03": 3, "age04": 4, "age07": 7, "age09": 9,
               "age10": 10, "age12": 12, "age14": 14, "age16": 16,
               "teds21": 21, "catslife": 26},
    "TRAILS": [11, 13.5],
    "CATSS":  [9, 12],
    "BHRC":   [9, 12, 15, 22],
    "cVEDA":  [14.5],
    "Gen R":  [13],
}

# ── Cohorts ────────────────────────────────────────────────────────────────────
# Palette and order lifted from plotting_predict_cognition.R so this figure and
# the planned-coverage figure are visually consistent.
COHORT_ORDER = ["ABCD", "ALSPAC", "Imagen", "TEDS",
                "TRAILS", "CATSS", "BHRC", "cVEDA", "Gen R"]

COHORT_COLOURS = {
    "ABCD":   "#1b9e77", "ALSPAC": "#d95f02", "Imagen": "#7570b3",
    "TEDS":   "#e7298a", "TRAILS": "#66a61e", "CATSS":  "#e6ab02",
    "BHRC":   "#a6761d", "cVEDA":  "#666666", "Gen R":  "#1f78b4",
}

# Folder/filename token -> display name used above.
DISPLAY_NAME = {
    "ABCD": "ABCD", "ALSPA": "ALSPAC", "ALSPAC": "ALSPAC",
    "IMAGEN": "Imagen", "Imagen": "Imagen",
    "TEDS": "TEDS", "TRAILS": "TRAILS", "CATSS": "CATSS",
    "BHRC": "BHRC", "CVEDA": "cVEDA", "cVEDA": "cVEDA",
    "GENR": "Gen R", "GenR": "Gen R",
}
FALLBACK_COLOURS = ["#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]

# ── Wave ordering ──────────────────────────────────────────────────────────────
# session_id is spelled differently in every cohort (wave0_arm_1, ses-00A, BL,
# FU1 ...). Resolve to an integer so waves sort and align across cohorts; fall
# back to alphabetical position when there is no number to find.
WAVE_ALIASES = {"bl": 0, "baseline": 0, "base": 0, "w0": 0, "t0": 0,
                "fu1": 1, "fu2": 2, "fu3": 3, "fu4": 4}


def wave_index(session_id) -> float:
    s = str(session_id).strip().lower()
    if s in WAVE_ALIASES:
        return float(WAVE_ALIASES[s])
    m = re.search(r"\d+", s)
    return float(m.group()) if m else float("nan")


def display_name(token: str) -> str:
    return DISPLAY_NAME.get(token, DISPLAY_NAME.get(token.upper(), token))


def resolve_cohort(path: str):
    """Work out which cohort a harmonised CSV belongs to.

    Filenames are not uniform across cohorts: BHRC writes
    harmonised_cognition_BHRC.csv, ABCD writes
    harmonised_cognition_v4_ABCDv7.csv. Stripping the prefix and trusting the
    rest gives "v4_ABCDv7", which is not a cohort, so the file would plot as a
    tenth unknown cohort with no colour and then be dropped by the R script.

    Three attempts, most specific first:
      1. the token after the prefix, as-is            (harmonised_cognition_TEDS)
      2. a known cohort name appearing anywhere in it (..._v4_ABCDv7 -> ABCD)
      3. the directory above output/                  (FINAL_OUTPUTS/ABCD/output/)
    Returns None if all three fail, so the caller can say so rather than
    silently inventing a cohort.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    token = stem.replace("harmonised_cognition_", "")

    if token in DISPLAY_NAME or token.upper() in DISPLAY_NAME:
        return display_name(token)

    # Longest match wins, so ALSPAC is never read as ALSPA.
    up = token.upper()
    hits = sorted((k for k in DISPLAY_NAME if k.upper() in up),
                  key=len, reverse=True)
    if hits:
        return DISPLAY_NAME[hits[0]]

    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent.lower() == "output":
        gp = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.abspath(path))))
        if gp in DISPLAY_NAME or gp.upper() in DISPLAY_NAME:
            return display_name(gp)

    return None


# ── Synthetic preview data ─────────────────────────────────────────────────────
def make_synthetic(tmp_dir: str) -> str:
    """Fake cohorts with realistic, deliberately ragged shapes.

    Different wave counts, different absent domains, very different sizes, one
    cohort carrying two columns for a domain -- so the layout is stress-tested
    on the awkward cases rather than a tidy rectangle.
    """
    rng = np.random.default_rng(0)
    # Session labels mirror each cohort's REAL spelling, because the age lookup
    # and the wave filters key off them. Using one generic scheme here made the
    # preview quietly drop whichever cohorts had keyed age tables.
    spec = {   # token: (n, session_ids, domains, single-wave domains)
        "ABCD":   (11874, [f"ses-0{i}A" for i in range(8)],
                   ["IQ", "RespInhib", "CogFlex", "WorkingMem", "Attention"], ["IQ"]),
        "ALSPA":  (15247, [f"wave{i}_arm_1" for i in range(5)],
                   ["IQ", "RespInhib", "CogFlex", "WorkingMem", "Attention"], []),
        "IMAGEN": (2223,  ["BL", "FU2", "FU3"],
                   ["IQ", "RespInhib", "CogFlex", "WorkingMem", "Attention"], []),
        "TEDS":   (16810, ["age02", "age04", "age07", "age09", "age10",
                           "age12", "age14", "age16", "teds21"],
                   ["IQ", "WorkingMem", "Attention"], []),
        "BHRC":   (2511,  [f"wave{i}_arm_1" for i in range(3)],
                   ["IQ", "RespInhib", "WorkingMem", "Attention"], ["IQ"]),
    }
    mid = {"IQ": "raw", "RespInhib": "comerr", "CogFlex": "computed",
           "WorkingMem": "back", "Attention": "rtvar"}

    for token, (n, sessions, doms, single) in spec.items():
        rows = []
        for w, sess in enumerate(sessions):
            retention = 1.0 if w == 0 else float(np.clip(0.90 ** w, 0.28, 1.0))
            for i in range(n):
                row = {"participant_id": f"{token}_{i:05d}",
                       "session_id": sess}
                for d in doms:
                    # TEDS: Working Memory and Attention exist at age16 only.
                    if token == "TEDS" and d in ("WorkingMem", "Attention") \
                            and sess != "age16":
                        val = np.nan
                    elif d in single and w > 0:
                        val = np.nan
                    else:
                        cover = retention * (0.55 if d == "Attention" else 0.87)
                        val = rng.normal() if rng.random() < cover else np.nan
                    row[f"{d}_{mid[d]}_notz"] = val
                # ABCD gets a second, sparser Response Inhibition column so the
                # "more than one measure -> take the biggest N" rule is exercised.
                if token == "ABCD" and "RespInhib" in doms:
                    row["RespInhib_alt_notz"] = (rng.normal()
                                                 if rng.random() < 0.25 * retention
                                                 else np.nan)
                rows.append(row)
        out = os.path.join(tmp_dir, token, "output")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(out, f"harmonised_cognition_{token}.csv"),
            index=False, na_rep="n/a")
    return tmp_dir


# ── Load + count ───────────────────────────────────────────────────────────────
def load_age_table(csv_dir: str, token: str, cohort: str):
    """Returns {session_id: age} from a wave_ages_<COHORT>.csv, or None.

    Written by get_wave_ages.py, which reads each cohort's own source data. Only
    aggregate statistics are stored in these files.
    """
    for folder in (csv_dir, _HERE):
        for name in (token, cohort, cohort.replace(" ", "")):
            path = os.path.join(folder, f"wave_ages_{name}.csv")
            if not os.path.exists(path):
                continue
            t = pd.read_csv(path)
            if AGE_TABLE_COL not in t.columns or "session_id" not in t.columns:
                print(f"  [warn] {path} lacks session_id/{AGE_TABLE_COL} -- ignored")
                continue
            return dict(zip(t["session_id"].astype(str), t[AGE_TABLE_COL]))
    return None


COUNTS_COLS = ["cohort", "domain", "session_id", "wave", "age", "n"]

# ── What each cohort actually uses ─────────────────────────────────────────────
# measures_used.csv is transcribed from the "Recommended variables" table at the
# top of each cohort's .md. It decides two things the data cannot:
#
#   metric    which measure to count when a cohort has several for one domain.
#             The old rule was "biggest N", which disagrees with the docs --
#             IMAGEN's Matrix Reasoning beats the recommended 4-subtest
#             composite by 4 participants, and TEDS's Raven's is a different
#             call again. Counting the recommended column means each domain is
#             represented by the measure the project actually uses.
#   sessions  which waves to count; BLANK MEANS ALL, which is the right answer
#             for every cohort today and the deliberate default. These figures
#             show what data EXISTS, so a wave a cohort's doc sets aside for
#             analytic reasons (IMAGEN Attention's FU2, say) still belongs in a
#             coverage picture. The column exists for a wave that is not the
#             same construct at all; the only such case, TEDS IQ below age 7,
#             is handled by WAVE_FILTERS because it is an age rule.
#
# A cohort or domain with no row here falls back to the old biggest-N rule and
# all waves, and says so on stdout -- so a newly harmonised cohort still plots,
# it is just flagged as undocumented.
SPEC_PATH = os.path.join(_HERE, "measures_used.csv")


def load_measure_spec(path: str) -> dict:
    """{(cohort, domain_key): {"metric": str|None, "sessions": set|None}}."""
    if not os.path.exists(path):
        print(f"  [WARN] {os.path.basename(path)} not found -- falling back to "
              f"biggest-N selection and all waves for every cohort.")
        return {}

    label_to_key = {v: k for k, v in DOMAIN_LABEL.items()}
    spec = {}
    df = pd.read_csv(path, dtype=str).fillna("")
    for _, r in df.iterrows():
        cohort = display_name(r["cohort"].strip())
        dom = label_to_key.get(r["domain"].strip())
        if dom is None:
            print(f"  [WARN] {os.path.basename(path)}: unknown domain "
                  f"'{r['domain']}' for {cohort} -- row ignored.")
            continue
        sess = [s.strip() for s in r.get("sessions", "").split(";") if s.strip()]
        spec[(cohort, dom)] = {
            "metric":   r.get("metric", "").strip() or None,
            "sessions": set(sess) if sess else None,
        }
    return spec


def merge_counts(local: pd.DataFrame, paths: list) -> pd.DataFrame:
    """Fold in counts CSVs computed in another environment.

    ABCD's harmonised data cannot leave its TRE, but the COUNTS can: they are
    aggregate (cohort, domain, wave, age, n) with no row per participant. So
    --counts-only is run inside the TRE, the resulting CSV is exported through
    the usual output check, and this merges it back in here.

    A cohort appearing both locally and in a merged file is taken from the
    merged file, which is the more deliberate of the two.
    """
    if not paths:
        return local

    frames = [local]
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"--merge-counts file not found: {p}")
        extra = pd.read_csv(p)
        missing = [c for c in COUNTS_COLS if c not in extra.columns]
        if missing:
            sys.exit(f"{p} is not a counts CSV -- missing column(s): "
                     f"{', '.join(missing)}\n"
                     f"  expected: {', '.join(COUNTS_COLS)}")
        cohorts = sorted(extra["cohort"].unique())
        clash = sorted(set(cohorts) & set(frames[0]["cohort"].unique()))
        if clash:
            print(f"  [merge] {os.path.basename(p)}: {', '.join(cohorts)} "
                  f"(replacing local {', '.join(clash)})")
            frames[0] = frames[0][~frames[0]["cohort"].isin(clash)]
        else:
            print(f"  [merge] {os.path.basename(p)}: {', '.join(cohorts)}")
        frames.append(extra[COUNTS_COLS])

    return pd.concat(frames, ignore_index=True)


def gather_inputs(root: str, explicit: list) -> list:
    """[(cohort, path)] from explicit --input entries, else from the glob."""
    pairs = []

    for item in explicit:
        if "=" in item and not os.path.exists(item):
            label, path = item.split("=", 1)
            cohort = display_name(label.strip())
        else:
            path = item
            cohort = resolve_cohort(path)
            if cohort is None:
                sys.exit(f"Cannot tell which cohort this file belongs to:\n"
                         f"  {path}\n"
                         f"  Name it explicitly, e.g. --input ABCD={path}")
        if not os.path.exists(path):
            sys.exit(f"--input file not found: {path}")
        pairs.append((cohort, path))

    if pairs:
        return pairs

    files = sorted(glob.glob(os.path.join(root, "*", "output",
                                          "harmonised_cognition_*.csv")))
    for path in files:
        cohort = resolve_cohort(path)
        if cohort is None:
            print(f"  [skip] {os.path.basename(path)}: cannot identify the "
                  f"cohort from the filename or its folder. Pass it as "
                  f"--input COHORT={path}")
            continue
        pairs.append((cohort, path))
    return pairs


def collect_counts(root: str) -> pd.DataFrame:
    pairs = gather_inputs(root, args.input)
    if not pairs:
        if args.allow_no_local or args.merge_counts:
            print("  (no harmonised CSVs found locally; "
                  "relying on --merge-counts)")
            return pd.DataFrame(columns=COUNTS_COLS)
        sys.exit(f"No harmonised CSVs found under {root}\n"
                 f"  expected <COHORT>/output/harmonised_cognition_*.csv\n"
                 f"  or pass one directly with --input [COHORT=]PATH")

    records = []
    for cohort, path in pairs:
        if os.path.basename(path) != f"harmonised_cognition_{cohort}.csv":
            print(f"      {cohort}: reading {os.path.basename(path)}")

        # Header first, so only the needed columns are loaded -- these files are
        # large and every z-score column is irrelevant here.
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "session_id" not in header:
            print(f"  [skip] {cohort}: no session_id column.")
            continue

        candidates: dict = {}
        for c in header:
            m = RAW_COL_RE.match(c)
            if m:
                candidates.setdefault(m.group(1), []).append(c)
        if not candidates:
            print(f"  [skip] {cohort}: no *_notz domain columns found.")
            continue

        # Prefer a wave_ages_<COHORT>.csv from get_wave_ages.py, then an age
        # column inside the harmonised CSV, then the nominal table.
        age_table = load_age_table(os.path.dirname(path), cohort, cohort)
        age_col = next((c for c in AGE_COL_CANDIDATES if c in header), None)
        all_cols = [c for v in candidates.values() for c in v]
        df = pd.read_csv(path,
                         usecols=["session_id"] + all_cols
                                 + ([age_col] if age_col else []),
                         na_values="n/a", low_memory=False)

        # Median age per wave -- an aggregate, never an individual value.
        if age_table is not None:
            age_by_wave = age_table
            AGE_SOURCE[cohort] = "observed (wave_ages table)"
        elif age_col:
            age_by_wave = df.groupby("session_id")[age_col].median().to_dict()
            AGE_SOURCE[cohort] = "observed (age column in CSV)"
        else:
            age_by_wave = None
            AGE_SOURCE[cohort] = "nominal"
        print(f"      {cohort} age source: {AGE_SOURCE[cohort]}")

        # Which measure, and which waves -- from the cohort's documentation
        # where it exists, biggest-N and all waves where it does not.
        chosen, keep_sessions = {}, {}
        for dom, cols in candidates.items():
            rule = SPEC.get((cohort, dom))
            keep_sessions[dom] = rule["sessions"] if rule else None

            if rule and rule["metric"]:
                want = f"{dom}_{rule['metric']}_notz"
                if want in cols:
                    chosen[dom] = want
                    if len(cols) > 1:
                        others = ", ".join(c for c in cols if c != want)
                        print(f"      {cohort} {dom}: documented metric "
                              f"-> {want}; not used: {others}")
                else:
                    print(f"      [WARN] {cohort} {dom}: measures_used.csv asks "
                          f"for '{want}' but the harmonised CSV has "
                          f"{', '.join(cols)} -- check the metric token. "
                          f"Falling back to biggest N.")

            if dom not in chosen:
                if len(cols) == 1:
                    chosen[dom] = cols[0]
                else:
                    ns = {c: int(df[c].notna().sum()) for c in cols}
                    best = max(ns, key=ns.get)
                    chosen[dom] = best
                    others = ", ".join(f"{c} n={ns[c]:,}"
                                       for c in cols if c != best)
                    print(f"      [undocumented] {cohort} {dom}: {len(cols)} "
                          f"measures -> biggest N {best} (n={ns[best]:,}); "
                          f"not used: {others}")

        absent = [d for d in DOMAINS if d not in chosen]
        print(f"  {cohort:<8} {len(df):>8,} rows   {len(chosen)}/{len(DOMAINS)} domains"
              + (f"   (absent: {', '.join(DOMAIN_LABEL[d] for d in absent)})" if absent else ""))

        for dom, col in chosen.items():
            g = df.groupby("session_id")[col].apply(lambda s: int(s.notna().sum()))

            keep = keep_sessions.get(dom)
            if keep:
                dropped = [s for s in g.index if s not in keep and g[s] > 0]
                missing = sorted(keep - set(g.index))
                if dropped:
                    print(f"      [doc] {cohort} {dom}: keeping "
                          f"{len(keep & set(g.index))} documented wave(s); "
                          f"dropped {', '.join(map(str, dropped))}")
                if missing:
                    print(f"      [WARN] {cohort} {dom}: measures_used.csv "
                          f"lists wave(s) absent from the CSV: "
                          f"{', '.join(missing)}")
                g = g[[s for s in g.index if s in keep]]

            for sess, n in g.items():
                records.append({"cohort": cohort, "domain": dom,
                                "session_id": sess, "wave": wave_index(sess),
                                "age": (age_by_wave.get(sess, np.nan)
                                        if age_by_wave else np.nan),
                                "n": int(n)})

    counts = pd.DataFrame(records)
    if counts.empty:
        sys.exit("No countable data found.")

    # Unnumbered session labels: fall back to alphabetical order within cohort.
    for coh, sub in counts.groupby("cohort"):
        if sub["wave"].isna().any():
            order = {s: i for i, s in enumerate(sorted(sub["session_id"].unique()))}
            mask = (counts["cohort"] == coh) & counts["wave"].isna()
            counts.loc[mask, "wave"] = counts.loc[mask, "session_id"].map(order)

    if not args.keep_empty_waves:
        total = counts.groupby(["cohort", "session_id"])["n"].transform("sum")
        for _, r in counts[total == 0][["cohort", "session_id"]].drop_duplicates().iterrows():
            print(f"  [drop] {r['cohort']}: {r['session_id']} empty for every domain")
        counts = counts[total > 0]

    # Drop zero counts outright. A domain measured at one wave only (IQ in most
    # cohorts) otherwise draws a line plunging to zero across the later waves,
    # which reads as catastrophic attrition rather than "not administered".
    # Removing them leaves a single marker, which is the honest picture.
    counts = counts[counts["n"] > 0]

    # Fill ages for cohorts whose CSV carried no age column, from NOMINAL_AGES
    # by wave position. Flagged loudly: these x-positions are planned ages, not
    # measured ones.
    nominal_used = []
    for coh in sorted(counts["cohort"].unique()):
        mask = (counts["cohort"] == coh) & counts["age"].isna()
        if not mask.any():
            continue
        table = NOMINAL_AGES.get(coh)
        if not table:
            print(f"  [WARN] {coh}: no age column and no NOMINAL_AGES entry -- "
                  f"cannot place on an age axis; add one to plot it.")
            continue
        if isinstance(table, dict):
            # Keyed by the cohort's own session_id -- the right form when waves
            # are not a contiguous 0,1,2,... run (IMAGEN's BL/FU2/FU3).
            counts.loc[mask, "age"] = counts.loc[mask, "session_id"].astype(str).map(
                lambda s: table.get(s, np.nan))
            missing = sorted(set(counts.loc[mask & counts["age"].isna(),
                                            "session_id"].astype(str)))
            if missing:
                print(f"  [WARN] {coh}: NOMINAL_AGES has no entry for "
                      f"session_id(s) {missing} -- add them, or those waves "
                      f"are dropped from the age axis.")
        else:
            waves_here = sorted({int(w) for w in counts.loc[mask, "wave"]})
            if waves_here and waves_here != list(range(len(waves_here))):
                print(f"  [WARN] {coh}: waves are numbered {waves_here}, not a "
                      f"contiguous 0..n run, but NOMINAL_AGES[{coh!r}] is a "
                      f"POSITIONAL list -- ages will be mis-assigned. Use a "
                      f"dict keyed by session_id instead.")
            counts.loc[mask, "age"] = counts.loc[mask, "wave"].map(
                lambda w: table[int(w)] if int(w) < len(table) else np.nan)
            short = sorted({int(w) for w in
                            counts.loc[mask & counts["age"].isna(), "wave"]})
            if short:
                print(f"  [WARN] {coh}: NOMINAL_AGES has {len(table)} entries but "
                      f"wave(s) {short} exist -- extend it.")
        nominal_used.append(coh)

    if nominal_used and args.x == "age":
        print("\n  " + "!" * 68)
        print("  AGE AXIS IS NOMINAL for: " + ", ".join(nominal_used))
        print("  These cohorts' CSVs carry no age column, so x-positions come")
        print("  from the planned-age table at the top of this script, NOT from")
        print("  the data. Counts (y) are real; ages (x) are approximate.")
        print("  Fix properly by adding an `age` column to the harmonised output.")
        print("  " + "!" * 68)
    AGE_SOURCE["_nominal_list"] = nominal_used

    counts = counts.dropna(subset=["age"]) if args.x == "age" else counts

    for (coh, dom), rule in WAVE_FILTERS.items():
        sel = (counts["cohort"] == coh) & (counts["domain"] == dom)
        if not sel.any():
            continue
        drop = sel & (counts["age"] < rule["min_age"]) if "min_age" in rule else sel & False
        if "max_age" in rule:
            drop = drop | (sel & (counts["age"] > rule["max_age"]))
        if drop.any():
            gone = sorted(set(counts.loc[drop, "session_id"].astype(str)))
            print(f"  [filter] {coh} {DOMAIN_LABEL.get(dom, dom)}: dropped "
                  f"{len(gone)} wave(s) outside the usable age range — "
                  f"{', '.join(gone)}")
            counts = counts[~drop]

    return counts.sort_values(["cohort", "domain", "wave"]).reset_index(drop=True)


# ── Plot ───────────────────────────────────────────────────────────────────────
AGE_SOURCE: dict = {}       # cohort -> "observed" | "nominal"


def ordered_cohorts(counts: pd.DataFrame):
    present = set(counts["cohort"])
    ranked = [c for c in COHORT_ORDER if c in present]
    return ranked + sorted(present - set(ranked))


def marker_for(cohort: str, used: list) -> str:
    """Marker shape per cohort, stable for the nine known cohorts."""
    if cohort in COHORT_MARKERS:
        return COHORT_MARKERS[cohort]
    m = FALLBACK_MARKERS[len(used) % len(FALLBACK_MARKERS)]
    used.append(m)
    return m


def assign_colours(cohorts) -> dict:
    colours, extra = {}, 0
    for c in cohorts:
        if c in COHORT_COLOURS:
            colours[c] = COHORT_COLOURS[c]
        else:
            colours[c] = FALLBACK_COLOURS[extra % len(FALLBACK_COLOURS)]
            extra += 1
    return colours


def resolve_break(counts: pd.DataFrame):
    """Return (low, high) for a broken count axis, or None for a continuous one.

    'auto' finds the widest empty band between consecutive observed counts and
    splits there -- but only if that band is a big enough share of the range to
    be worth the visual cost of a break.
    """
    spec = (args.ybreak or "none").strip().lower()
    if spec in ("", "none", "off", "0"):
        return None
    vals = np.sort(counts["n"].unique().astype(float))
    if spec == "auto":
        if len(vals) < 4:
            print("  [ybreak] too few distinct counts to split -- continuous axis.")
            return None
        gaps = np.diff(vals)
        k = int(np.argmax(gaps))
        lo, hi = vals[k], vals[k + 1]
        span = vals[-1] - vals[0]
        if span <= 0 or (hi - lo) / span < 0.25:
            print("  [ybreak] no gap wide enough to justify a break -- "
                  "continuous axis.")
            return None
        lo, hi = lo * 1.08, hi * 0.92
        print(f"  [ybreak] auto split at {lo:,.0f} / {hi:,.0f}")
        return (lo, hi)
    try:
        lo, hi = (float(x) for x in spec.replace(" ", "").split(","))
    except ValueError:
        sys.exit(f"--ybreak expects 'none', 'auto', or LOW,HIGH -- got {args.ybreak!r}")
    if hi <= lo:
        sys.exit("--ybreak: HIGH must be greater than LOW.")
    return (lo, hi)


def nice_step(span: float, target: int = 5) -> float:
    """A round tick interval giving roughly `target` ticks across `span`."""
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def build_single_panel(counts: pd.DataFrame) -> plt.Figure:
    """All domains on one axes.

    COLOUR = cognitive domain, MARKER SHAPE = cohort, every line solid. A shared
    y-axis is the point: it makes counts comparable ACROSS domains, not just
    across cohorts within a domain.

    With --ybreak the count axis is split into two stacked panels sharing one x,
    so a 16,000-participant cohort cannot flatten a 2,000-participant one into
    the baseline. The two panels are NOT on the same scale -- that is the whole
    device -- so a line's gradient is only comparable within a panel.
    """
    cohorts = ordered_cohorts(counts)
    _used: list = []
    markers = {c: marker_for(c, _used) for c in cohorts}
    panels = [d for d in DOMAINS if d in set(counts["domain"])]
    xcol = "age" if args.x == "age" else "wave"
    brk = resolve_break(counts)

    # Domains within a cohort are measured at the SAME visit, so their points
    # land on identical x and the thick lines hide one another. Fan them out by
    # a fixed per-domain offset, centred on zero so the group still straddles
    # the true age. Deterministic, not random.
    jit_unit = args.jitter if args.x == "age" else args.jitter * 0.25
    n_pan = max(len(panels), 1)

    def x_of(series, dom):
        off = (panels.index(dom) - (n_pan - 1) / 2) * jit_unit
        return series[xcol] + off

    if brk:
        # Height each panel in proportion to the span it shows, so both end up
        # with the SAME counts-per-inch. That keeps one tick interval sensible
        # across the break and, more usefully, makes a line's gradient mean the
        # same thing in both panels.
        _top = counts["n"].max() * 1.04
        _hi_span = max(_top - brk[1], 1.0)
        _lo_span = max(brk[0], 1.0)
        _h = 1.4 + 7.4 * (_hi_span + _lo_span) / max(_hi_span + _lo_span, 1.0)
        fig, (ax_hi, ax_lo) = plt.subplots(
            2, 1, sharex=True, figsize=(14.5, 9.8),
            gridspec_kw={"height_ratios": [_hi_span, _lo_span], "hspace": 0.17})
        draw_axes = [ax_hi, ax_lo]
        ax_main = ax_lo
    else:
        fig, ax = plt.subplots(figsize=(14.5, 8.6))
        draw_axes = [ax]
        ax_hi, ax_lo, ax_main = None, ax, ax

    # Everything is drawn on BOTH panels; each panel's ylim decides what shows.
    for ax in draw_axes:
        for coh in cohorts:
            for dom in panels:
                s = (counts[(counts["cohort"] == coh) & (counts["domain"] == dom)]
                     .sort_values(xcol))
                if s.empty:
                    continue
                xs = x_of(s, dom)
                ax.plot(xs, s["n"], linestyle="-", marker=markers[coh],
                        markersize=MARKER_SIZE * MARKER_SHAPE_SCALE.get(
                            markers[coh], 1.0),
                        linewidth=LINE_WIDTH,
                        color=DOMAIN_COLOURS.get(dom, "#555555"),
                        alpha=0.95, zorder=3, solid_capstyle="round",
                        markeredgecolor="white", markeredgewidth=1.4)
                if args.annotate:
                    for (_, r), xj in zip(s.iterrows(), xs):
                        ax.annotate(f"{int(r['n']):,}", (xj, r["n"]),
                                    textcoords="offset points", xytext=(0, 12),
                                    ha="center", fontsize=7.5, fontweight="bold",
                                    color=DOMAIN_COLOURS.get(dom, "#555555"))

        if args.log:
            ax.set_yscale("log")
        ax.grid(linestyle=":", linewidth=0.8, color="#d4d4d4", zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
            sp.set_linewidth(1.2)
        ax.tick_params(labelsize=11.5, width=1.2, length=5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    if brk:
        lo_cut, hi_cut = brk
        top = counts["n"].max() * 1.04
        ax_hi.set_ylim(hi_cut, top)
        ax_lo.set_ylim(0, lo_cut)

        # One tick interval for both panels, so the y labels read consistently
        # and equal vertical distances mean equal numbers of participants.
        step = nice_step(max(top - hi_cut, lo_cut), target=5)
        for _a in (ax_hi, ax_lo):
            _a.yaxis.set_major_locator(mticker.MultipleLocator(step))

        # Hide the facing spines; the break marks themselves are drawn AFTER
        # tight_layout, since their position is read from the final axes boxes.
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(labelbottom=False, bottom=False)
    elif not args.log:
        ax_main.set_ylim(bottom=0)

    for ax in draw_axes:
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")

    ax_main.set_xlabel("Age (years)" if args.x == "age" else "Wave",
                       fontsize=14, fontweight="bold", labelpad=9)
    for lbl in ax_main.get_xticklabels():
        lbl.set_fontweight("bold")
    if args.x == "age":
        xlo = int(np.floor(counts["age"].min() / 2) * 2)
        xhi = int(np.ceil(counts["age"].max() / 2) * 2)
        ax_main.set_xticks(np.arange(xlo, xhi + 1, 2))
    else:
        ax_main.set_xticks(sorted(counts["wave"].unique()))

    fig.supylabel("Participants with usable data", fontsize=14,
                  fontweight="bold", x=0.055)

    # Two legends: colour carries the domain, marker shape carries the cohort.
    dom_handles = [Line2D([0], [0], color=DOMAIN_COLOURS[d], linewidth=6.0,
                          label=DOMAIN_LABEL[d]) for d in panels]
    coh_handles = [Line2D([0], [0], color="#444444", linestyle="-",
                          linewidth=3.0, marker=markers[c],
                          markersize=12.5 * MARKER_SHAPE_SCALE.get(markers[c], 1.0),
                          markeredgecolor="white", markeredgewidth=1.4,
                          label=c) for c in cohorts]

    leg1 = fig.legend(handles=dom_handles, title="Cognitive domain",
                      loc="upper left", bbox_to_anchor=(0.995, 0.92),
                      frameon=False, fontsize=11.5, title_fontsize=12.5)
    leg1.get_title().set_fontweight("bold")
    fig.add_artist(leg1)

    leg2 = fig.legend(handles=coh_handles, title="Cohort", loc="lower left",
                      bbox_to_anchor=(0.995, 0.06), frameon=False,
                      fontsize=11.5, title_fontsize=12.5)
    leg2.get_title().set_fontweight("bold")

    n_sub = 1 + bool(args.jitter) + bool(brk) + bool(
        (AGE_SOURCE.get("_nominal_list") or []) and args.x == "age")
    head = 0.055 + 0.022 * n_sub          # figure fraction reserved for titles
    fig.suptitle("Cognitive Domain Coverage Across Cohorts",
                 fontsize=17, fontweight="bold", y=0.995, va="top")
    sub = ("Participants with usable (non-missing) harmonised data.  "
           "Colour = domain, marker shape = cohort.")
    if args.jitter:
        sub += ("\nDomains are offset horizontally by a fixed "
                f"{args.jitter:g} y so overlapping lines stay visible — "
                "points sit beside, not at, the exact age.")
    if brk:
        sub += (f"\nNote the broken count axis: 0–{brk[0]:,.0f} below, "
                f"{brk[1]:,.0f}+ above. Both panels share one scale and tick "
                "interval, so gradients stay comparable.")
    nominal = AGE_SOURCE.get("_nominal_list") or []
    if nominal and args.x == "age":
        sub += ("\nAge positions are NOMINAL (planned) for: "
                + ", ".join(nominal) + " \u2014 counts are real, ages approximate.")
    fig.text(0.5, 0.995 - 0.038, sub, ha="center", va="top", fontsize=9.5,
             color="#555555", linespacing=1.45)
    fig.tight_layout(rect=[0.03, 0, 1, 1 - head])

    # Break marks last: get_position() is only final once tight_layout has run.
    if brk:
        d = 0.011
        kw = dict(transform=fig.transFigure, color="#333333",
                  linewidth=1.5, clip_on=False)
        y_hi = ax_hi.get_position().y0
        y_lo = ax_lo.get_position().y1
        for xp in (ax_lo.get_position().x0, ax_lo.get_position().x1):
            fig.add_artist(plt.Line2D([xp - d, xp + d], [y_hi - d, y_hi + d], **kw))
            fig.add_artist(plt.Line2D([xp - d, xp + d], [y_lo - d, y_lo + d], **kw))
    return fig


def build_stacked(counts: pd.DataFrame) -> plt.Figure:
    """ONE panel: a stacked bar per year, segments coloured by cognitive domain.

    Each segment is the total number of participants with usable data for that
    domain at that age, summed across every cohort. Complements the line figure:
    that one asks "how is each cohort's coverage changing", this one asks "how
    much data of each kind exists at this age".

    Colour carries the domain, matching the line figure exactly. Cohort identity
    is deliberately absent -- that is what the other figure is for.

    ONE IMPORTANT CAVEAT, stated on the figure itself: the segments sum ACROSS
    DOMAINS, and the same participant contributes to several of them (a child
    with IQ, working memory and attention data is counted three times). The bar
    total is therefore a count of available MEASUREMENTS, not of people. Segment
    heights are the honest numbers; read the total as a rough volume indicator.
    """
    w = max(args.bin_width, 0.1)
    d = counts.copy()
    if args.x == "age":
        # A bar labelled 9 spans ages 9.0-9.9, drawn centred on 9.5.
        d["edge"] = np.floor(d["age"] / w) * w
        d["bin"] = d["edge"] + w / 2
    else:
        d["edge"] = d["wave"]
        d["bin"] = d["wave"]
        w = 1.0

    per = d.groupby(["domain", "bin", "edge"])["n"].sum().reset_index()
    doms = [dom for dom in DOMAINS if dom in set(per["domain"])]
    bins = sorted(per["bin"].unique())
    edges = sorted(per["edge"].unique())

    fig, ax = plt.subplots(figsize=(15.5, 8.8))

    # Bars only. A line tracing each domain's segment top was tried and dropped:
    # the top of a stacked segment moves with everything beneath it, so the lines
    # read as trends that are not there.
    bottom = np.zeros(len(bins))
    for dom in doms:
        vals = (per[per["domain"] == dom].set_index("bin")["n"]
                .reindex(bins, fill_value=0).values.astype(float))
        ax.bar(bins, vals, width=w * 0.86, bottom=bottom,
               color=DOMAIN_COLOURS.get(dom, "#999999"),
               edgecolor="white", linewidth=0.9, zorder=3,
               label=DOMAIN_LABEL.get(dom, dom))
        bottom = bottom + vals

    totals = bottom
    for b, t in zip(bins, totals):
        if t > 0:
            ax.text(b, t * 1.012, f"{int(t):,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlabel("Age (years)" if args.x == "age" else "Wave",
                  fontsize=14, fontweight="bold", labelpad=9)
    ax.set_ylabel("Participants with usable data", fontsize=14,
                  fontweight="bold", labelpad=9)
    ax.set_ylim(0, max(totals.max() * 1.1, 1))
    if args.x == "age":
        ax.set_xticks([e + w / 2 for e in edges])
        ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, color="#d4d4d4", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
        sp.set_linewidth(1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11.5, width=1.2, length=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    handles = [Line2D([0], [0], color=DOMAIN_COLOURS[dom], linewidth=9,
                      label=DOMAIN_LABEL.get(dom, dom)) for dom in doms]
    leg = fig.legend(handles=handles, title="Cognitive domain", loc="center left",
                     bbox_to_anchor=(1.0, 0.5), frameon=False,
                     fontsize=11.5, title_fontsize=12.5)
    leg.get_title().set_fontweight("bold")

    fig.suptitle("Total Cognitive Data Available by Age",
                 fontsize=17, fontweight="bold", y=0.995, va="top")
    sub_txt = (f"Participants with usable data, summed across ALL cohorts in "
               f"{w:g}-year bins (a bar labelled 9 spans ages 9.0-9.9).\n"
               "Segment height = that domain's total.\n"
               "Bar totals count MEASUREMENTS, not people — one participant "
               "contributes to every domain they completed.")
    nominal = AGE_SOURCE.get("_nominal_list") or []
    if nominal and args.x == "age":
        sub_txt += ("\nAge positions are NOMINAL (planned) for: "
                    + ", ".join(nominal) + ".")
    fig.text(0.5, 0.955, sub_txt, ha="center", va="top", fontsize=9.5,
             color="#555555", linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    return fig


def build_figure(counts: pd.DataFrame) -> plt.Figure:
    cohorts = ordered_cohorts(counts)
    colours = assign_colours(cohorts)
    panels = [d for d in DOMAINS if d in set(counts["domain"])]
    xcol = "age" if args.x == "age" else "wave"
    waves = sorted(counts[xcol].unique())

    # Reserve a fixed strip of HEIGHT (not a fraction) for the title block, so
    # the two lines never collide regardless of how many panels are drawn.
    panel_h, title_h = 2.45, 1.15
    fig_h = panel_h * len(panels) + title_h
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, fig_h),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, dom in zip(axes, panels):
        sub = counts[counts["domain"] == dom]

        if args.style == "line":
            for coh in cohorts:
                s = sub[sub["cohort"] == coh].sort_values(xcol)
                if s.empty:
                    continue
                ax.plot(s[xcol], s["n"],
                        marker="o", markersize=6.0, linewidth=2.6,
                        color=colours[coh], alpha=1.0,
                        label=coh, zorder=3, solid_capstyle="round")
                if args.annotate:
                    for _, r in s.iterrows():
                        ax.annotate(f"{int(r['n']):,}", (r[xcol], r["n"]),
                                    textcoords="offset points", xytext=(0, 7),
                                    ha="center", fontsize=6.5, color=colours[coh])
        else:
            present = [c for c in cohorts if not sub[sub["cohort"] == c].empty]
            width = 0.82 / max(len(present), 1)
            for k, coh in enumerate(present):
                s = sub[sub["cohort"] == coh].set_index(xcol)["n"]
                xs = np.array(waves, float) + (k - (len(present) - 1) / 2) * width
                ax.bar(xs, [s.get(w, 0) for w in waves], width=width * 0.9,
                       color=colours[coh], label=coh, zorder=3,
                       edgecolor="white", linewidth=0.4)

        ax.set_ylabel("Participants", fontsize=9)
        ax.set_title(DOMAIN_LABEL.get(dom, dom), fontsize=12,
                     fontweight="bold", loc="left", pad=5, color="#111111")
        if args.log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#d9d9d9", zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
            sp.set_linewidth(0.9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.tick_params(labelsize=9)

    axes[-1].set_xlabel("Age (years)" if args.x == "age" else "Wave", fontsize=11)
    if args.x == "age":
        lo = int(np.floor(min(waves) / 2) * 2); hi = int(np.ceil(max(waves) / 2) * 2)
        axes[-1].set_xticks(np.arange(lo, hi + 1, 2))
    else:
        axes[-1].set_xticks(waves)
        axes[-1].set_xticklabels([str(int(w)) for w in waves])

    handles = [Line2D([0], [0], color=colours[c], linewidth=3.0,
                      marker="o" if args.style == "line" else "s",
                      markersize=6, label=c)
               for c in cohorts]
    leg = fig.legend(handles=handles, title="Cohorts", loc="center left",
                     bbox_to_anchor=(1.005, 0.5), frameon=False,
                     fontsize=11.5, title_fontsize=12.5)
    leg.get_title().set_fontweight("bold")

    top = 1 - title_h / fig_h          # panels end here; titles live above
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.suptitle("Cognitive Domain Coverage Across Cohorts",
                 fontsize=17, fontweight="bold",
                 y=1 - 0.22 * title_h / fig_h, va="top")
    fig.text(0.5, 1 - 0.62 * title_h / fig_h,
             "Participants with usable (non-missing) harmonised data, by wave.\n"
             "A cohort absent from a panel has no measure for that domain; a "
             "single point means the domain was measured at one wave only.",
             ha="center", va="top", fontsize=9.5, color="#555555",
             linespacing=1.45)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
if args.synthetic:
    import tempfile
    tmp = tempfile.mkdtemp(prefix="synthetic_cohorts_")
    print("SYNTHETIC MODE -- numbers below are FAKE, for layout tuning only.")
    print(f"Fake data in {tmp}; FINAL_OUTPUTS is neither read nor written.\n")
    root = make_synthetic(tmp)
else:
    root = args.final_outputs
    print(f"Reading cohorts under {root}")

SPEC = load_measure_spec(SPEC_PATH)
if SPEC:
    print(f"Measure/wave spec: {os.path.basename(SPEC_PATH)} "
          f"({len(SPEC)} cohort x domain rows)")

print("\nCohorts found:")
counts = collect_counts(root)
counts = merge_counts(counts, args.merge_counts)

if counts.empty:
    sys.exit("No counts from any source -- nothing to do.")

# Written before any figure, so --counts-only can stop immediately after and so
# a crash in the drawing code never costs the counting work.
out_csv = os.path.join(OUT_DIR, "coverage_across_cohorts.csv")
counts.to_csv(out_csv, index=False)

print("\nParticipants with non-missing data, by wave:")
wide = (counts.pivot_table(index=["domain", "cohort"], columns="wave",
                           values="n", fill_value=0)
        .reindex([d for d in DOMAINS if d in set(counts["domain"])], level=0))
print(wide.to_string())

if args.counts_only:
    print(f"\nSaved -> {out_csv}")
    print("Aggregate counts only: cohort, domain, session_id, wave, age, n.\n"
          "No individual-level data and no row per participant -- this is the\n"
          "file to put through the output check and carry out.")
    print("\nDone (--counts-only: no figures drawn).")
    sys.exit(0)

if not _HAVE_MPL:
    sys.exit("matplotlib is not installed, so no figure can be drawn.\n"
             f"  The counts were still written to {out_csv}.\n"
             "  Re-run with --counts-only to suppress this, or draw the "
             "figures elsewhere from that CSV.")

suffix = ("_facet" if args.layout == "facet" else "") \
    + ("_bar" if args.style == "bar" else "") \
    + ("_bywave" if args.x == "wave" else "") \
    + ("_log" if args.log else "")

written = []
if args.figure in ("lines", "both"):
    fig = (build_single_panel(counts) if args.layout == "single"
           else build_figure(counts))
    out_png = os.path.join(OUT_DIR, f"coverage_across_cohorts{suffix}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(out_png)

if args.figure in ("stacked", "both"):
    fig = build_stacked(counts)
    out_png = os.path.join(OUT_DIR,
                           f"coverage_stacked_across_cohorts{suffix}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(out_png)

out_png = written[-1]

for _p in written:
    print(f"\nSaved -> {_p}")
print(f"Saved -> {out_csv}  (aggregate counts only -- safe to share)")
print("\nDone.")
