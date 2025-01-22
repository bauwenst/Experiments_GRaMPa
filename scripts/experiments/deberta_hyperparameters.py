from scripts.preamble import *
from scripts.constants import *

from typing import Type

from transformers import DebertaConfig, DebertaForMaskedLM
from transformers import AutoTokenizer

from lamoto.tasks import MlmHyperparameters, SUGGESTED_HYPERPARAMETERS_MLM
from lamoto.training.auxiliary.hyperparameters import getDefaultHyperparameters, Intervals, EveryNDescents, AfterNDescents
from lamoto.training.lineages import SerialisedTokeniser, ConfigFactory, BaseModel
from archit.instantiation.heads import MaskedLMHeadConfig
from tktkt.util.environment import IS_NOT_LINUX


class DebertaConfigFactory(ConfigFactory):
    def buildConfig(self, serial_tokeniser: SerialisedTokeniser, base_model: Type[BaseModel]) -> DebertaConfig:
        config = DebertaConfig.from_pretrained("microsoft/deberta-base")  # IMPORTANT NOTE: initialising with DebertaConfig() actually **DISABLES** relative attention. The docs lie. Initialising from deberta-base enables it.
        H = DEBERTA_HIDDEN_SIZE
        config.hidden_size = H
        config.intermediate_size   = H*4    # https://arxiv.org/pdf/1908.08962
        config.num_attention_heads = H//64  # https://arxiv.org/pdf/1908.08962

        print("\nInstantiating tokeniser to get vocab size...")
        config.vocab_size = len(AutoTokenizer.from_pretrained(serial_tokeniser).get_vocab()) if isinstance(serial_tokeniser, (str,Path)) else serial_tokeniser.buildTokeniser().getVocabSize()
        config.num_hidden_layers = DEBERTA_LAYERS
        config.max_relative_positions = DEBERTA_K
        config.tie_word_embeddings = True

        config.max_position_embeddings = DEBERTA_CONTEXT_LIMIT  # (Note: the 1024 you see in the relative positional embeddings matrix is 2*k, not this number, see docs of DebertaConfig.) The whole point of the paper is "help this tokeniser has a lot of tokens", so if we're going to truncate it better be decently far out. Can't be too far nevertheless, because pushing a full context through should still be feasible (in fact, that's what happens in packing). The use of relative embeddings means that we don't need to worry about the exact number and could arguably alter it at test time.
        return config


def getPretrainingHyperparameters() -> MlmHyperparameters:
    hp = SUGGESTED_HYPERPARAMETERS_MLM.copy()
    hp.SEED = 69420
    hp.store_in_hf_cache = True

    # Set lineage fields to None.
    hp.MODEL_CONFIG_OR_CHECKPOINT = None
    hp.archit_basemodel_class = None
    hp.TOKENISER = None

    # Model setup
    hp.init_weights = False
    hp.custom_hf_class = DebertaForMaskedLM  # Although this class cannot correctly load the Microsoft checkpoints, it can train and load checkpoints from scratch.
    hp.archit_head_config = MaskedLMHeadConfig()

    # Device-specific
    if IS_NOT_LINUX:
        hp.EXAMPLES_PER_DEVICEBATCH = 2
    else:
        hp.WANDB_PROJECT = WANDB_PROJECT
        hp.EXAMPLES_PER_DEVICEBATCH = 128  # Even when packing 1024 tokens per example, this fits on an A100. 94% VRAM usage though. Tight.
        hp.EXAMPLES_PER_EFFECTIVE_BATCH = 2048  # As in DeBERTa paper and also the best in the RoBERTa paper.

    # Training parameters
    hp.learning_rate = 1e-3  # DeBERTa uses 2e-4. I use 5x that because small learning rates are dangerous as we saw in the GPT experiment for HEL.
    hp.EFFECTIVE_BATCHES_WARMUP = 256  # We bank on 2 days of fine-tuning with top speed being 24 batches/hr, so 1.2k total. 12-layer models do 10k warmup steps... I wouldn't go lower than 256: https://arxiv.org/abs/2406.09405v1
    hp.MLM_PROBABILITY = 0.15

    hp.HARD_STOPPING_CONDITION = AfterNDescents(descents=512)  # Should be easily reachable within 2 days on an A100. SP ULM can do over 1000 in 40 hours. HF BPE can do 900 in 40 hours. GRaMPa can get to 512 in a little over 30 hours.

    # Testing parameters
    hp.EXAMPLES_PER_EVALUATION = 2**14  # 16k examples, or 8 batches. Our fast tokenisers do 150s/b = 2.5m/b and so 20m for eval. Our slow tokenisers need 220s = 3.67m/b, i.e. 1.5x the time and hence 30m for eval.
    hp.TRACK_BEST_MODEL = True
    hp.EVAL_VS_SAVE_INTERVALS = Intervals(
        # evaluation=EveryNMinutes(minutes=9*20),  # Train for 9 x 20 minutes (3 hours, equalling 72 train batches), then evaluate for 1 x 20 minutes. Hence, exactly 10% of compute is spent verifying that we don't overfit. Every save cycle is 200 minutes = 3h20m. For the slow tokeniser, it's 30m for an eval and 9 x 20 == 6 x 30 so 1/7 = 14% of compute in eval, with 3 hours being 48 batches.
        evaluation=EveryNDescents(descents=64),  # Makes more sense than minutes for two reasons: (1) you can compare measurements between models independent of tokeniser speed, and (2) unlike time intervals, the fraction of time spent in compute vs. spent in eval is a constant.
        checkpointing=None,
        backups=EveryNDescents(descents=256)
    )
    return hp
