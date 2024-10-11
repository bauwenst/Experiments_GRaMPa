#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=48:00:00

cd $VSC_DATA/Python/ExternalRepos/WiaT/
conda activate draft
PYTHONPATH=. python tst/experiments/deberta_pretraining.py --low_resource --model_id 1
