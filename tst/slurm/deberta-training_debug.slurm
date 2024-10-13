#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=16
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=00:25:00

cd $VSC_DATA/Python/ExternalRepos/WiaT/
conda activate draft
nvidia-smi
PYTHONPATH=. python tst/experiments/deberta_pretraining.py --low_resource --model_id 1
