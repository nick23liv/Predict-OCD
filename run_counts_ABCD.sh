#!/bin/bash
# =============================================================================
# run_counts_ABCD.sh
# Slurm job script for the ABCD TRE
#
# Submit with:   submit_job run_counts_ABCD.sh
# (NOT sbatch — the TRE requires the submit_job wrapper)
#
# Counts how many participants have usable data per cognitive domain per wave,
# from the harmonised ABCD output, so ABCD can appear in the cross-cohort
# coverage figures without its individual-level data leaving the TRE.
#
# DISCLOSURE CONTROL: the only output is coverage_across_cohorts.csv, which
# holds one row per cohort x domain x wave with n and a median age. No
# participant identifier, no individual score, no row per participant. This is
# aggregate-only and is the file intended for egress; nothing else from this job
# should leave. Nothing is plotted here — the figures are drawn outside the TRE
# from this CSV.
#
# After the job completes, retrieve the output from:
#   /hpc_jobs/<your-k-number>/job_<slurm_job_id>/
# =============================================================================

#SBATCH --job-name=cog_counts_abcd
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/job_scratch/cog_counts_abcd_%j.out
#SBATCH --error=/job_scratch/cog_counts_abcd_%j.err

# ── Echo job info for the log ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job name      : ${SLURM_JOB_NAME}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "============================================"

# ── Paths ──────────────────────────────────────────────────────────────────────
# EDIT THIS if you move plot_coverage_across_cohorts.py. Taken from the path in
# the traceback when it was run interactively.
SCRIPT_DIR="${HOME}/Documents/Wellcome_OCD/cognition/scripts/Predict-OCD"
SCRIPT="${SCRIPT_DIR}/plot_coverage_across_cohorts.py"

# The harmonised ABCD output. The script works out that this is ABCD from the
# filename, so it does not need renaming.
INPUT="${SCRIPT_DIR}/output/harmonising/harmonised_cognition_v4_ABCDv7.csv"

# Outputs go to /job_scratch; retrievable afterwards from
# /hpc_jobs/<k-number>/job_<SLURM_JOB_ID>/
OUT_DIR="/job_scratch"

# ── Activate virtual environment ──────────────────────────────────────────────
# Same venv as run_get_wave_ages_ABCD.sh; module is not available on this TRE.
# --counts-only needs pandas and numpy only: it draws nothing, and the script
# imports matplotlib lazily so a venv without it still works.
source "${HOME}/venvs/harmonise/bin/activate"
echo "Python      : $(which python)"
python -c "import pandas, numpy; print('pandas', pandas.__version__, '| numpy', numpy.__version__)"

# ── Verify the script and input exist before running ──────────────────────────
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: script not found: ${SCRIPT}"
    echo "       Edit SCRIPT_DIR at the top of this file."
    exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
    echo "ERROR: missing input file: ${INPUT}"
    exit 1
fi

# measures_used.csv is OPTIONAL but decides which measure represents a domain
# where ABCD has several. It must sit beside the script. Without it the job
# still runs and falls back to whichever measure has the biggest N, saying so.
if [[ ! -f "${SCRIPT_DIR}/measures_used.csv" ]]; then
    echo "NOTE: measures_used.csv not found beside the script."
    echo "      Measure choice will fall back to biggest-N and will be logged"
    echo "      as [undocumented]. Copy it across if that matters."
fi

# A wave_ages_ABCD.csv beside the script or beside the input makes the age axis
# observed rather than planned. run_get_wave_ages_ABCD.sh produces it.
if [[ -f "${SCRIPT_DIR}/wave_ages_ABCD.csv" ]]; then
    echo "Found wave_ages_ABCD.csv — ages will be observed, not nominal."
else
    echo "NOTE: no wave_ages_ABCD.csv beside the script, so ages will be"
    echo "      NOMINAL (planned). Counts are real either way. To fix, copy the"
    echo "      output of run_get_wave_ages_ABCD.sh into ${SCRIPT_DIR}."
fi

echo "Script and input found."

# ── Run ────────────────────────────────────────────────────────────────────────
echo "Counting ABCD cognitive coverage …"
python "${SCRIPT}" \
    --counts-only \
    --input   "${INPUT}" \
    --out-dir "${OUT_DIR}"

EXIT_CODE=$?

# ── Finish ─────────────────────────────────────────────────────────────────────
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Output CSV : ${OUT_DIR}/coverage_across_cohorts.csv"
    echo "Retrieve from: /hpc_jobs/<your-k-number>/job_${SLURM_JOB_ID}/"
    echo
    echo "Check two things in this log before requesting egress:"
    echo "  1. the cohort reads as 'ABCD', not 'v4_ABCDv7'"
    echo "  2. any '[undocumented] ABCD <domain>' lines — these name the measure"
    echo "     counted where ABCD has several. If one disagrees with the ABCD"
    echo "     .md, set the 'metric' column in measures_used.csv and re-run."
fi
echo "============================================"

exit ${EXIT_CODE}
