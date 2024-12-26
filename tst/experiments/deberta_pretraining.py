from tst.preamble import *
from tst.constants import *
from tst.experiments.tokenisers_instances import *

from transformers import PreTrainedTokenizerBase
from transformers.models.deberta.modeling_deberta import DebertaForMaskedLM, DebertaConfig

from archit.instantiation.basemodels import DebertaBaseModel
from lamoto.tasks import MLM_SlimPajama, SUGGESTED_HYPERPARAMETERS_MLM
from lamoto.tasks.mlm import MaskedLMHeadConfig, MLM_C4
from lamoto.training.auxiliary.hyperparameters import Intervals, EveryNMinutes, EveryNDescents, AfterNDescents
from lamoto.augmenting.augment_dataset import TaskWithAugmentedDataset, Truncate
from tktkt.util.environment import IS_NOT_LINUX
from tktkt.interfaces.huggingface import TktktToHuggingFace


def makeConfig(tk: PreTrainedTokenizerBase) -> DebertaConfig:
    config = DebertaConfig.from_pretrained("microsoft/deberta-base")
    H = DEBERTA_HIDDEN_SIZE
    config.hidden_size = H
    config.intermediate_size   = H*4    # https://arxiv.org/pdf/1908.08962
    config.num_attention_heads = H//64  # https://arxiv.org/pdf/1908.08962

    config.num_hidden_layers = DEBERTA_LAYERS
    config.max_relative_positions = DEBERTA_K
    config.vocab_size = len(tk.get_vocab())
    config.tie_word_embeddings = True

    config.max_position_embeddings = DEBERTA_CONTEXT_LIMIT  # (Note: the 1024 you see in the relative positional embeddings matrix is 2*k, not this number, see docs of DebertaConfig.) The whole point of the paper is "help this tokeniser has a lot of tokens", so if we're going to truncate it better be decently far out. Can't be too far nevertheless, because pushing a full context through should still be feasible (in fact, that's what happens in packing). The use of relative embeddings means that we don't need to worry about the exact number and could arguably alter it at test time.
    return config


def deberta_pretraining(tk: PreTrainedTokenizerBase, tk_name: str, low_resource: bool, continue_from_checkpoint: str=None):
    hp = SUGGESTED_HYPERPARAMETERS_MLM

    hp.SEED = 69420
    hp.SAVE_AS = "deberta" + "-" + tk_name + "_low"*low_resource

    # These are only to parse the config. We are going to use an HF class since LaMoTO has no weight tying yet.
    hp.archit_basemodel_class = DebertaBaseModel
    hp.archit_head_config = MaskedLMHeadConfig()

    # Model setup
    hp.init_weights = False
    hp.custom_hf_class = DebertaForMaskedLM  # Although this class cannot correctly load the Microsoft checkpoints, it can train and load checkpoints from scratch.
    hp.MODEL_CONFIG_OR_CHECKPOINT = makeConfig(tk)

    # Tokeniser
    hp.TOKENISER = tk

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
        #evaluation=EveryNMinutes(minutes=9*20),  # Train for 9 x 20 minutes (3 hours, equalling 72 train batches), then evaluate for 1 x 20 minutes. Hence, exactly 10% of compute is spent verifying that we don't overfit. Every save cycle is 200 minutes = 3h20m. For the slow tokeniser, it's 30m for an eval and 9 x 20 == 6 x 30 so 1/7 = 14% of compute in eval, with 3 hours being 48 batches.
        evaluation=EveryNDescents(descents=64),  # Makes more sense than minutes for two reasons: (1) you can compare measurements between models independent of tokeniser speed, and (2) unlike time intervals, the fraction of time spent in compute vs. spent in eval is a constant.
        checkpointing=None
    )

    ###
    if IS_NOT_LINUX:  # SlimPajama takes too long to get a stream for
        task = MLM_C4(packing=True)
    else:
        task = MLM_SlimPajama(packing=True)

    if low_resource:
        task = TaskWithAugmentedDataset(task, augmentation=Truncate(max_examples=50_000), splits={"train"})
    task.train(hp, resume_from_folder=Path(continue_from_checkpoint) if continue_from_checkpoint else None)


if __name__ == "__main__":
    if IS_NOT_LINUX:
        model_id = 1
        low_res = True
        continue_from = None
    else:
        # Get model ID and low-resourcedness from command line
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=int)
        parser.add_argument("--low_resource", action="store_true")
        parser.add_argument("--continue_from", type=str)
        args = parser.parse_args()

        model_id = args.model_id
        low_res = args.low_resource
        continue_from = args.continue_from

    # This is all you need.
    tokeniser, shorthand = getTokeniserByModelId(model_id)
    deberta_pretraining(
        TktktToHuggingFace(tokeniser),
        tk_name=shorthand,
        low_resource=low_res,
        continue_from_checkpoint=continue_from
    )
