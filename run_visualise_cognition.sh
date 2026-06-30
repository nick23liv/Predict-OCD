#!/bin/bash
# Submit with:  submit_job run_visualize_cognition.sh
# Outputs retrievable from:  /hpc_jobs/<your-k-number>/job_<SLURM_JOB_ID>/

#SBATCH --job-name=visualize_cognition
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/job_scratch/visualize_cognition_%j.out
#SBATCH --error=/job_scratch/visualize_cognition_%j.err

module load python/3.9   # adjust to match available version: check with `module avail python`

# Activate venv if needed (see run_harmonise_cognition.sh for setup instructions)
# source "${HOME}/venvs/harmonise/bin/activate"

python "${HOME}/harmonise_cognition/visualize_cognition.py" \
    --input   "${HOME}/harmonise_cognition/harmonised_cognition_ABCDv7.csv" \
    --dyn     "/dataset/abcd/v7/phenotype/ab_g_dyn.tsv" \
    --out-dir "/job_scratch"

echo "Exit code: $?  |  Retrieve figures from /hpc_jobs/<k-number>/job_${SLURM_JOB_ID}/"
