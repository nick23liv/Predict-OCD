#!/bin/bash
# =========================================================
# Count ABCD cognitive coverage inside the TRE.
#
# Produces coverage_across_cohorts.csv -- AGGREGATE ONLY
# (cohort, domain, session_id, wave, age, n). No individual-level data and no
# row per participant, so it is the file to put through the output check and
# carry out. Nothing is drawn here; the figures are made outside the TRE from
# that CSV.
#
#   sbatch run_counts_slurm.sh
#   squeue -u $USER
#   cat counts_abcd_<jobid>.log
# =========================================================
#SBATCH --job-name=cog_counts
#SBATCH --output=counts_abcd_%j.log
#SBATCH --error=counts_abcd_%j.log
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Your site may require these -- copy the values from the submission script that
# ran the harmonisation, and delete the leading "#" to enable.
##SBATCH --partition=cpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

# ── 1. EDIT THIS: paths ───────────────────────────────────
# Where this script and plot_coverage_across_cohorts.py live.
SCRIPT_DIR="$HOME/Documents/Wellcome_OCD/cognition/scripts/Predict-OCD"
# The harmonised ABCD output.
INPUT="$SCRIPT_DIR/output/harmonising/harmonised_cognition_v4_ABCDv7.csv"
# Where the counts CSV should land.
OUT_DIR="$SCRIPT_DIR/output/counts"

# ── 2. EDIT THIS: the Python environment ──────────────────
# The node needs pandas and numpy. matplotlib is NOT required: --counts-only
# draws nothing, and the script imports matplotlib lazily for exactly this case.
#
# Use whatever the harmonisation job used. Uncomment ONE of these, or add your
# own. If you are unsure, `cat` the submission script that produced
# harmonised_cognition_v4_ABCDv7.csv and copy its environment lines here.

# -- a) Lmod / Environment Modules
# module purge
# module load Python/3.11.3-GCCcore-12.3.0
# module load SciPy-bundle          # on many sites this is what carries numpy/pandas

# -- b) A virtualenv
# source "$HOME/venvs/cognition/bin/activate"

# -- c) Conda
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate cognition

# -- d) A full interpreter path, no activation needed
PYTHON=python3

# ── 3. Run ────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

echo "Host:   $(hostname)"
echo "Python: $(command -v "$PYTHON")"
"$PYTHON" -c "import numpy, pandas; print('numpy', numpy.__version__, '/ pandas', pandas.__version__)"
echo

"$PYTHON" plot_coverage_across_cohorts.py \
    --counts-only \
    --input "$INPUT" \
    --out-dir "$OUT_DIR"

echo
echo "================================================================"
echo "Counts written to: $OUT_DIR/coverage_across_cohorts.csv"
echo "Aggregate only -- this is the file to request out of the TRE."
echo "================================================================"

# Check the log for two things before exporting:
#   'ABCD: reading harmonised_cognition_v4_ABCDv7.csv' and a cohort named ABCD
#      -- not 'v4_ABCDv7', which would mean the name was not recognised.
#   '[undocumented] ABCD <domain>: N measures -> biggest N ...'
#      -- where ABCD has several measures for one domain, this says which was
#         counted. If it disagrees with the ABCD .md, fill in the `metric`
#         column of measures_used.csv for ABCD and re-run.
