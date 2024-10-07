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

cd $VSC_DATA/Python/ExternalRepos/WiaT/
conda activate draft
PYTHONPATH=. python tst/experiments/hyperparameters.py --experiment_multiplex --fixed_temp 2.0 --bpe_vocab
