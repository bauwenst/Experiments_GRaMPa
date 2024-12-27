#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=16
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=48:00:00

# Below, the model ID is filled by the number you put after the script when you submit it.
# E.g.: sbatch deberta-training.slurm 6  will put the value 6 into the variable $1.

cd $VSC_DATA/Python/ExternalRepos/Experiments_GRaMPa/
conda activate draft
nvidia-smi
PYTHONPATH=. python tst/experiments/deberta_pretraining.py --low_resource --model_id $1
