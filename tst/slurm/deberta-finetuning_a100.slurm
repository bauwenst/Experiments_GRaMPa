#!/bin/bash -l
#SBATCH --account=lp_lagom
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=18
#SBATCH --mail-user=thomas.bauwens@kuleuven.be
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=01:00:00

cd $VSC_DATA/Python/ExternalRepos/WiaT/
conda activate draft
nvidia-smi
PYTHONPATH=. python tst/experiments/deberta_finetuning.py --n_samples 5 --max_batches_at_bs32 512 --checkpoint $VSC_DATA/Python/ExternalRepos/WiaT/data/out/lamoto/checkpoints/$1 --old_model_id $2 --task_id $3 --typo_id 1
