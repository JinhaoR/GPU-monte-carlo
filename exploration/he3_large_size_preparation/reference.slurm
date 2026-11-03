#!/bin/bash
# Requirements example: add Kraken account/partition/GPU-type names locally.
#SBATCH --job-name=he3-L256-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@1800
#SBATCH --output=he3-reference-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Activate the SAME tested Python/CUDA environment before this command.
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Validate in a short interactive allocation first; see REPORT.md.
# No automatic requeue or campaign loop. Re-submit this one case to continue.
srun python -u large_run.py
