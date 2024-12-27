#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --mem-per-cpu=28000M
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00

cd $VSC_DATA/Python/ExternalRepos/Experiments_GRaMPa/
conda activate draft
PYTHONPATH=. python scripts/experiments/train_tokenisers.py
