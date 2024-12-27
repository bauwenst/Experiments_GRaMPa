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

cd $VSC_DATA/Python/ExternalRepos/Experiments_GRaMPa/
conda activate draft
nvidia-smi
PYTHONPATH=. python tst/experiments/run.py --lineage $1 --node $2 --n_samples ${3-0} --n_32batches_phase1 ${4-0}
