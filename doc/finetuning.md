# Notes on finetuning
We take 5 hyperparameter grid samples and take the best of those. 
The best sample sometimes has a suspiciously heightened eval loss, based on which we know we should take more samples.
(The associated test loss is also unusually low, but the only signal that matters to the experimenter is the eval loss.)

We reran the following tunings:
- NER:
  - GRaMPa+BPE l=1 t=1.0
  - GRaMPa+BPE l=1 t=-10.0

- DP:
  - GRaMPa+BPE l=2 t=1.0

- DP train typos:
  - GRaMPa+BPE l=2 t=1
  - GRaMPa+BPE l=2 t=-10

- DP test typos:
  - GRaMPa+BPE l=1 t=1

- SST-2:
  - GRaMPa+ULM l=1 t=5.0

- QQP:
  - GRaMPa+ULM l=1 t=5.0
  - GRaMPa+BPE l=1 t=5.0
  - GRaMPa+BPE l=1 t=-10.0

- RTE:
  - GRaMPa+BPE l=1 t=-10.0
  - GRaMPa+ULM l=1 t=-10.0

```
sbatch run_h100.slurm 13 sst2 10 512
sbatch run_h100.slurm 9 ner 10 512
sbatch run_h100.slurm 11 ner 10 512
sbatch run_h100.slurm 3 dp 10 512
sbatch run_h100.slurm 13 qqp 10 512
sbatch run_h100.slurm 10 qqp 10 512
sbatch run_h100.slurm 11 qqp 10 512
sbatch run_h100.slurm 11 rte 10 512
sbatch run_h100.slurm 14 rte 10 512

sbatch run_h100.slurm 3 dp-typos-2 10 512
sbatch run_h100.slurm 5 dp-typos-2 10 512
sbatch run_h100.slurm 9 dp-typos-1 10 512
```
