#!/bin/bash
#----------------------------------------------------------------------------
# TACC batch submission for the full Riemannian sweep (6 arms x 28 pairs).
#
#   sbatch submit_tacc.slurm
#   squeue -u $USER
#   tail -f $SCRATCH/FineMI-enhancements/Riemannian/results/logs/sweep.log
#
# EDIT BEFORE SUBMITTING: -A (allocation), -p (queue), -t (walltime), --mail-user.
#   Lonestar6  -p gpu-a100      (3x A100 40GB per node)  max 48h
#              -p gpu-a100-dev  (development)            max  2h
# Run on the SAME GPU model as the FBCNet sweep (A100) so the reused raw baseline is
# bit-identical to what this node would produce.
#----------------------------------------------------------------------------
#SBATCH -J fine-riemann
#SBATCH -o fine-riemann.o%j
#SBATCH -e fine-riemann.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

set -euo pipefail

module purge
module load gcc/11.2.0 2>/dev/null || true
module load cuda/12 2>/dev/null || true

# $SCRATCH, never $HOME, for job I/O.
REPO="$SCRATCH/FineMI-enhancements/Riemannian"
export FINE_DATASET_ROOT="$SCRATCH/FineMI"
export PY="$SCRATCH/venvs/fine/bin/python"      # same venv as the other techniques

cd "$REPO"
echo "host  : $(hostname)"
echo "jobid : ${SLURM_JOB_ID:-none}"
nvidia-smi -L

export SHARDS="${SHARDS:-$(nvidia-smi -L | wc -l)}"
srun ./run_riemannian.sh full

echo "done. results in $REPO/results"
