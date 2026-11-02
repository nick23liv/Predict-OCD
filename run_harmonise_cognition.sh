#!/bin/bash
# =============================================================================
# run_harmonise_cognition.sh
# Slurm job script for the ABCD TRE
#
# Submit with:   submit_job run_harmonise_cognition.sh
# (NOT sbatch — the TRE requires the submit_job wrapper)
#
# TIERED (master doc §2.4): this runs the harmonisation TWICE, once per tier.
#   --tier ml    harmonised_cognition_ABCDv7_ml.csv   17 cols, modelling input
#   --tier all   harmonised_cognition_ABCDv7_all.csv  20 cols, broad product
#                plus QC/evaluated_metrics_ABCDv7.csv — a WORKING FILE holding
#                the rejected Millisecond Flanker, for the _all figures only.
#                Do NOT copy it to FINAL_OUTPUTS and do not model from it.
#
# Every output is tier-suffixed, so the second run cannot overwrite the first.
#
# After the job completes, retrieve outputs from:
#   /hpc_jobs/<your-k-number>/job_<slurm_job_id>/
#
# THEN copy the two CSVs and the QC folder back to ${SCRIPT_DIR} below, because
# the visualisation job runs separately and cannot read this job's /job_scratch.
# =============================================================================

#SBATCH --job-name=harmonise_cognition
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/job_scratch/harmonise_cognition_%j.out
#SBATCH --error=/job_scratch/harmonise_cognition_%j.err

# ── Echo job info for the log ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job name      : ${SLURM_JOB_NAME}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "============================================"

# ── Paths ──────────────────────────────────────────────────────────────────────
# ABCD v7 phenotype data on the TRE (read-only)
DATA_DIR="/dataset/abcd/v7/phenotype"

# EDIT THIS if you put the script elsewhere. The filename must match what you
# copied across — this is the current script name, not the older
# harmonize_cognition.py.
SCRIPT_DIR="${HOME}/harmonise_cognition"
SCRIPT="${SCRIPT_DIR}/harmonising_cognition_v4_ABCDv7.py"

# All job outputs go to /job_scratch; retrievable after the job from
# /hpc_jobs/<k-number>/job_<SLURM_JOB_ID>/
OUT_DIR="/job_scratch"

# ── Activate virtual environment ──────────────────────────────────────────────
# module is not available on this TRE; Python 3.10 is on the system PATH.
# The venv below must be created once before submitting this job:
#
#   python -m venv ~/venvs/harmonise
#   source ~/venvs/harmonise/bin/activate
#   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
#       pandas numpy matplotlib scipy
#
source "${HOME}/venvs/harmonise/bin/activate"
echo "Python      : $(which python)"
python -c "import pandas, numpy; print('pandas', pandas.__version__, '| numpy', numpy.__version__)"

# ── Verify the script exists ──────────────────────────────────────────────────
if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: script not found: ${SCRIPT}"
    echo "       Edit SCRIPT_DIR / SCRIPT at the top of this file."
    exit 1
fi

# ── Verify the input files exist before running ───────────────────────────────
# nc_y_flnkr.tsv is required again as of v7: it carries the tier-'eval'
# Millisecond Flanker, which the 'all' run needs for its figures. The 'ml' run
# does not open it.
echo "Checking input files in: ${DATA_DIR}"
for f in ab_g_dyn.tsv nc_y_wisc.tsv nc_y_nihtb.tsv nc_y_lmt.tsv nc_y_flnkr.tsv; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        echo "ERROR: missing input file: ${DATA_DIR}/${f}"
        exit 1
    fi
done
echo "All input files found."

# ── Run both tiers ─────────────────────────────────────────────────────────────
EXIT_CODE=0
for TIER in ml all; do
    echo ""
    echo "────────────────────────────────────────────"
    echo "TIER '${TIER}'  —  $(date)"
    echo "────────────────────────────────────────────"
    python "${SCRIPT}" \
        --data-dir "${DATA_DIR}" \
        --tier     "${TIER}" \
        --output   "${OUT_DIR}/harmonised_cognition_ABCDv7_${TIER}.csv"
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
    echo "  harmonised_cognition_ABCDv7_ml.csv    (modelling input)"
    echo "  harmonised_cognition_ABCDv7_all.csv   (broad product)"
    echo "  QC/QC_report_ABCDv7_{ml,all}.txt      (full run transcripts)"
    echo "  QC/QC_*_ABCDv7_{ml,all}.png           (6 figures)"
    echo "  QC/evaluated_metrics_ABCDv7.csv       (WORKING FILE — not an output)"
    echo ""
    echo "Retrieve from: /hpc_jobs/<your-k-number>/job_${SLURM_JOB_ID}/"
    echo "Then copy the CSVs and QC/ to ${SCRIPT_DIR} before running the"
    echo "visualisation job, which cannot read this job's /job_scratch."
fi
echo "============================================"

exit ${EXIT_CODE}
