#!/bin/bash
#----------------------------------------------------------------------------
# TACC batch submission for the full 28-pair x 2-arm FINE sweep.
#
#   sbatch submit_tacc.slurm
#   squeue -u $USER
#   tail -f $SCRATCH/fine_results/logs/sweep.log
#
# EDIT BEFORE SUBMITTING: -A (allocation), -p (queue), -t (walltime).
# Queue names differ per system:
#   Lonestar6  -p gpu-a100        (3x A100 40GB per node)  max 48h
#              -p gpu-a100-dev    (development)            max  2h
#   Vista      -p gh              (GH200)
#   Frontera   -p rtx             (4x RTX 5000 per node)
# Check yours with:  sinfo -s     /  qlimits
#----------------------------------------------------------------------------
#SBATCH -J fine-sweep
#SBATCH -o fine-sweep.o%j
#SBATCH -e fine-sweep.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

set -euo pipefail

# ---- modules -------------------------------------------------------------
# Build the venv and download the dataset ONCE, wherever is convenient (this
# system's compute nodes have outbound network access, so either works). The
# batch job itself only computes.
module purge
module load gcc/11.2.0 2>/dev/null || true
module load cuda/12 2>/dev/null || true
module list 2>&1 || true

# ---- paths ---------------------------------------------------------------
# $SCRATCH, never $HOME: $HOME has a small quota and is not for job I/O.
REPO="$SCRATCH/FineMI-enhancements/morlet-scalogram"
export FINE_DATASET_ROOT="$SCRATCH/FineMI"
export FINE_RESULTS_ROOT="$SCRATCH/fine_results"
export PY="$SCRATCH/venvs/fine/bin/python"

cd "$REPO"

echo "host      : $(hostname)"
echo "jobid     : ${SLURM_JOB_ID:-none}"
echo "gpus      :"
nvidia-smi -L
echo "dataset   : $FINE_DATASET_ROOT"
echo "results   : $FINE_RESULTS_ROOT"

# One worker per GPU. Raise to 2x the GPU count if nvidia-smi shows the
# cards are not saturated (this model is small and may not fill an A100).
export SHARDS="${SHARDS:-$(nvidia-smi -L | wc -l)}"

srun ./run_sweep.sh full

echo "done. results in $FINE_RESULTS_ROOT"
