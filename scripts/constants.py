from dataclasses import dataclass

BPEDROPOUT_P = 0.1
ULM_K     = 64
ULM_ALPHA = 0.15

TYPO_P = 0.15

DEBERTA_HIDDEN_SIZE = 512
DEBERTA_LAYERS = 6
DEBERTA_K = 512  # Half of the full relative window
DEBERTA_CONTEXT_LIMIT = 1024


@dataclass
class _ExperimentsConfig:
    n_tuning_samples: int
    n_32batches_phase1: int


EXPERIMENT_CONFIG = _ExperimentsConfig(n_tuning_samples=5, n_32batches_phase1=512)
WANDB_PROJECT = "GRaMPa"
