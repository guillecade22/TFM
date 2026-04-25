#!/bin/bash
#SBATCH -J TFM_EEG_Diffusion          # Job name
#SBATCH -n 4                          # Number of CPU tasks (threads)
#SBATCH -N 1                          # Number of nodes
#SBATCH -D /hhome/ricse01/TFM/TFM     # Working directory
#SBATCH -t 0-24:00                    # Max runtime: 24 hours
#SBATCH -p dcca40                     # Partition/queue
#SBATCH --mem 16000                   # Memory (RAM)
#SBATCH --gres gpu:1                  # Request 1 GPU
#SBATCH -o slurm_io/%x_%u_%j.out      # stdout log
#SBATCH -e slurm_io/%x_%u_%j.err      # stderr log

set -euo pipefail

nvidia-smi

cd /hhome/ricse01/TFM/TFM/

export PYTHONPATH=/hhome/ricse01/TFM/TFM${PYTHONPATH:+:$PYTHONPATH}

source /hhome/ricse01/miniconda3/bin/activate

conda activate BCI

python Generation/Evaluate/RenconstructionMetricsSDXL.py
