#!/bin/bash
# =============================================================================
# run_get_wave_ages_ABCD.sh
# Slurm job script for the ABCD TRE
#
# Submit with:   submit_job run_get_wave_ages_ABCD.sh
# (NOT sbatch — the TRE requires the submit_job wrapper)
#
# Extracts per-wave age summary statistics from ab_g_dyn.tsv so the cross-cohort
# coverage figure can plot real ages instead of planned ones.
#
# DISCLOSURE CONTROL: the only output is wave_ages_ABCD.csv, which holds one row
# per session_id with n and summary statistics (mean/sd/median/p10/p90/min/max).
# No participant identifier, no individual age, no cognitive score. Waves with
# fewer than --min-n participants are dropped entirely. This is aggregate-only
# and is the file intended for egress; nothing else from this job should leave.
#
# After the job completes, retrieve the output from:
#   /hpc_jobs/<your-k-number>/job_<slurm_job_id>/
# =============================================================================

#SBATCH --job-name=get_wave_ages_abcd
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/job_scratch/get_wave_ages_abcd_%j.out
#SBATCH --error=/job_scratch/get_wave_ages_abcd_%j.err

# ── Echo job info for the log ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job name      : ${SLURM_JOB_NAME}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "============================================"

# ── Paths ──────────────────────────────────────────────────────────────────────
# ABCD v7 phenotype data on the TRE (read-only)
INPUT="/dataset/abcd/v7/phenotype/ab_g_dyn.tsv"

# EDIT THIS if you move get_wave_ages.py. This is the directory you were in when
# you ran it interactively.
SCRIPT_DIR="${HOME}/Documents/Wellcome_OCD/cognition/scripts/Predict-OCD/14Jul/Predict-OCD"
SCRIPT="${SCRIPT_DIR}/get_wave_ages.py"

# Outputs go to /job_scratch; retrievable afterwards from
# /hpc_jobs/<k-number>/job_<SLURM_JOB_ID>/
OUT_DIR="/job_scratch"

# Suppress any wave with fewer than this many aged participants. Stated
# explicitly rather than left to the script default so it appears in the log.
MIN_N=10

# ── Activate virtual environment ──────────────────────────────────────────────
# module is not available on this TRE; the venv carries numpy/pandas.
# Created once with:
#   python3 -m venv ~/venvs/harmonise
#   source ~/venvs/harmonise/bin/activate
#   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
#       pandas numpy matplotlib scipy
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
echo "Script and input found."

# ── Run ────────────────────────────────────────────────────────────────────────
echo "Running get_wave_ages.py for ABCD …"
python "${SCRIPT}" \
    --cohort  ABCD \
    --input   "${INPUT}" \
    --out-dir "${OUT_DIR}" \
    --min-n   "${MIN_N}"

EXIT_CODE=$?

# ── Finish ─────────────────────────────────────────────────────────────────────
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Output CSV : ${OUT_DIR}/wave_ages_ABCD.csv"
    echo "Retrieve from: /hpc_jobs/<your-k-number>/job_${SLURM_JOB_ID}/"
fi
echo "============================================"

exit ${EXIT_CODE}
