#!/bin/bash
# =============================================================================
# run_visualise_cognition.sh
# Slurm job script for the ABCD TRE
#
# Submit with:   submit_job run_visualise_cognition.sh
# (NOT sbatch — the TRE requires the submit_job wrapper)
#
# TIERED (master doc §2.4): runs the visualisation TWICE, once per tier.
#   --tier ml    5 metrics — the selected metric per domain
#   --tier all   7 metrics — adds IQ_scaled and the rejected Millisecond Flanker
#
# RUN THE HARMONISATION JOB FIRST, then copy its CSVs and QC/ folder into
# ${INPUT_DIR} below. /job_scratch is per-job, so this job cannot read the
# harmonisation job's scratch directory.
#
# After the job completes, retrieve figures from:
#   /hpc_jobs/<your-k-number>/job_<slurm_job_id>/
# =============================================================================

#SBATCH --job-name=visualise_cognition
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/job_scratch/visualise_cognition_%j.out
#SBATCH --error=/job_scratch/visualise_cognition_%j.err

# WALL TIME: the previous single-tier run took ~25 min for 4 slope metrics. This
# job now fits 4 on 'ml' and 5 on 'all', each against two mixed models, plus two
# Figure 1 renders. 3 hours is deliberate headroom — a model that fails to
# converge is far slower than one that does, and losing the run to the wall
# clock costs more than the queue time does.

# ── Echo job info for the log ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job name      : ${SLURM_JOB_NAME}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "============================================"

# ── Paths ──────────────────────────────────────────────────────────────────────
# EDIT THESE if you put things elsewhere. INPUT_DIR is where you copied the
# harmonisation job's outputs back to — it must contain
# harmonised_cognition_ABCDv7_ml.csv and QC/evaluated_metrics_ABCDv7.csv.
SCRIPT_DIR="${HOME}/harmonise_cognition"
SCRIPT="${SCRIPT_DIR}/visualise_cognition_LMM.py"
INPUT_DIR="${SCRIPT_DIR}"

DYN_FILE="/dataset/abcd/v7/phenotype/ab_g_dyn.tsv"
OUT_DIR="/job_scratch"

# ── Activate virtual environment ──────────────────────────────────────────────
# module is not available on this TRE; Python 3.10 is on the system PATH.
# Activate the venv created during setup (see run_harmonise_cognition.sh).
source "${HOME}/venvs/harmonise/bin/activate"
echo "Python      : $(which python)"
python -c "import pandas, numpy, statsmodels; print('statsmodels', statsmodels.__version__)"

# ── Verify the script and the design file ─────────────────────────────────────
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: script not found: ${SCRIPT}"
    echo "       Edit SCRIPT_DIR / SCRIPT at the top of this file."
    exit 1
fi
if [[ ! -f "${DYN_FILE}" ]]; then
    echo "ERROR: missing design file (needed for visit age): ${DYN_FILE}"
    exit 1
fi

# ── Run both tiers ─────────────────────────────────────────────────────────────
# The 'all' tier reads the QC evaluation working file, which is the only file
# carrying the rejected metric. If nothing was rejected it is not written at
# all, so fall back to the broad CSV.
ML_INPUT="${INPUT_DIR}/harmonised_cognition_ABCDv7_ml.csv"
ALL_INPUT="${INPUT_DIR}/QC/evaluated_metrics_ABCDv7.csv"
if [[ ! -f "${ALL_INPUT}" ]]; then
    ALL_INPUT="${INPUT_DIR}/harmonised_cognition_ABCDv7_all.csv"
fi

EXIT_CODE=0
for TIER in ml all; do
    if [[ "${TIER}" == "ml" ]]; then INPUT="${ML_INPUT}"; else INPUT="${ALL_INPUT}"; fi

    if [[ ! -f "${INPUT}" ]]; then
        echo "ERROR: missing input for tier '${TIER}': ${INPUT}"
        echo "       Run the harmonisation job first, then copy its CSVs and"
        echo "       QC/ folder into ${INPUT_DIR}."
        EXIT_CODE=1
        break
    fi

    echo ""
    echo "────────────────────────────────────────────"
    echo "TIER '${TIER}'  —  $(date)"
    echo "Input: ${INPUT}"
    echo "────────────────────────────────────────────"
    python "${SCRIPT}" \
        --tier    "${TIER}" \
        --input   "${INPUT}" \
        --dyn     "${DYN_FILE}" \
        --out-dir "${OUT_DIR}"
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "ERROR: tier '${TIER}' exited ${rc} — stopping."
        EXIT_CODE=${rc}
        break
    fi
done

# ── Finish ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Outputs in ${OUT_DIR}:"
    echo "  trajectories_ABCDv7_{ml,all}.png"
    echo "  same_vs_individual_slopes_ABCDv7_{ml,all}.png"
    echo "  trajectory_report_ABCDv7_{ml,all}.txt   (holds the dBIC verdict table)"
    echo ""
    echo "Retrieve from: /hpc_jobs/<your-k-number>/job_${SLURM_JOB_ID}/"
fi
echo "============================================"

exit ${EXIT_CODE}
