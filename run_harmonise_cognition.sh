#!/bin/bash
# =============================================================================
# run_harmonise_cognition.sh
# Slurm job script for KCL CREATE TRE HPC
#
# Submit with:   submit_job run_harmonise_cognition.sh
# (NOT sbatch — the TRE requires the submit_job wrapper)
#
# After the job completes, retrieve outputs from:
#   /hpc_jobs/<your-k-number>/job_<slurm_job_id>/
# =============================================================================

#SBATCH --job-name=harmonise_cognition
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
# ABCD v7 phenotype data directory on the TRE (read-only)
DATA_DIR="/dataset/abcd/v7/phenotype"

# Location of the Python script — update this to wherever you placed it
# (e.g. your home directory or a project folder on the TRE)
SCRIPT_DIR="${HOME}/harmonise_cognition"
SCRIPT="${SCRIPT_DIR}/harmonize_cognition.py"

# All job outputs go to /job_scratch; they are retrievable after the job
# from /hpc_jobs/<k-number>/job_<SLURM_JOB_ID>/
OUTPUT_FILE="/job_scratch/harmonised_cognition.csv"

# ── Load Python module ─────────────────────────────────────────────────────────
# Run `module avail python` on the TRE to confirm the exact module name;
# the examples below cover the most common names on CREATE HPC.
# Uncomment the one that matches your environment:
#
module load python/3.9          # adjust version as available
# module load python3/3.10
# module load anaconda3/2023.09  # if a conda-based module is the only option
#                                # (note: conda *environments* don't work, but
#                                #  the base pandas/numpy are usually available)

# ── Activate a virtual environment (if you have one) ──────────────────────────
# If pandas/numpy are already available via the module above, skip this block.
# If you need to install packages, set up the venv first (outside this script)
# with the trusted-host flags the TRE requires:
#
#   python -m venv ${HOME}/venvs/harmonise
#   source ${HOME}/venvs/harmonise/bin/activate
#   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
#       pandas numpy
#
# Then uncomment the activate line below:
# source "${HOME}/venvs/harmonise/bin/activate"

# ── Verify the input files exist before running ───────────────────────────────
echo "Checking input files in: ${DATA_DIR}"
for f in ab_g_dyn.tsv nc_y_wisc.tsv nc_y_flnkr.tsv nc_y_nihtb.tsv nc_y_lmt.tsv; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        echo "ERROR: missing input file: ${DATA_DIR}/${f}"
        exit 1
    fi
done
echo "All input files found."

# ── Run the harmonisation script ───────────────────────────────────────────────
echo "Running harmonise_cognition.py …"
python "${SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --output   "${OUTPUT_FILE}"

EXIT_CODE=$?

# ── Finish ─────────────────────────────────────────────────────────────────────
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Output CSV : ${OUTPUT_FILE}"
    echo "Retrieve from: /hpc_jobs/<your-k-number>/job_${SLURM_JOB_ID}/"
fi
echo "============================================"

exit ${EXIT_CODE}
