from tst.preamble import *

from transformers import PreTrainedTokenizerBase

from transformers.models.deberta.modeling_deberta import DebertaForMaskedLM, DebertaConfig

from wiat.training.archit_base import DebertaBaseModel
from lamoto.tasks import MLM_SlimPajama, SUGGESTED_HYPERPARAMETERS_MLM
from lamoto.tasks.mlm import MaskedLMHeadConfig
from tktkt.util.environment import IS_LINUX


def deberta_pretraining(tk: PreTrainedTokenizerBase):
    hp = SUGGESTED_HYPERPARAMETERS_MLM

    hp.SEED = 69420
    hp.SAVE_AS = "deberta"

    # These are only to parse the config. We are going to use a HF class since LaMoTO has no weight tying yet.
    hp.archit_basemodel_class = DebertaBaseModel
    hp.archit_head_config = MaskedLMHeadConfig()

    # Model setup
    hp.init_weights = False
    hp.custom_hf_class = DebertaForMaskedLM  # Although this class cannot correctly load the Microsoft checkpoints, it can train and load checkpoints from scratch.
    hp.MODEL_CONFIG_OR_CHECKPOINT = DebertaConfig.from_pretrained("microsoft/deberta-base")  # TODO: Configure this.

    # Tokeniser
    hp.TOKENISER = tk

    # Device-specific
    if IS_LINUX:
        hp.WANDB_PROJECT = "wiat"
        hp.EXAMPLES_PER_DEVICEBATCH = 64  # Should definitely fit on an A100.
    else:
        hp.EXAMPLES_PER_DEVICEBATCH = 64  # TODO: Adjust to your liking.

    # Training parameters
    hp.LEARNING_RATE = 5e-3  # DeBERTa uses 2e-4. I use 20x that because small learning rates are dangerous.
    hp.EFFECTIVE_BATCHES_WARMUP = 10_000
    hp.MLM_PROBABILITY = 0.15

    # Testing parameters
    hp.EXAMPLES_PER_EVALUATION = 2**14  # Two times the amount of data processed for one descent.

    ###
    task = MLM_SlimPajama(truncate_train_split_to=100_000)
    task.train(hp)


if __name__ == "__main__":
    from tktkt.interfaces.huggingface import TktktToHuggingFace
    from tst.experiments.tokenisers_instances import createTokeniser_SwitchyGrampa_ULM
    deberta_pretraining(TktktToHuggingFace(createTokeniser_SwitchyGrampa_ULM()))
