#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=12:00:00

cd $VSC_DATA/Python/ExternalRepos/Experiments_GRaMPa/
conda activate draft
PYTHONPATH=. python scripts/experiments/tokenisers_hyperparameters.py --experiment_temperature --bpe_vocab
