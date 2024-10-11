from tst.preamble import *

from transformers import PreTrainedTokenizerBase
from transformers.models.deberta.modeling_deberta import DebertaForMaskedLM, DebertaConfig

from wiat.training.archit_base import DebertaBaseModel
from lamoto.tasks import MLM_SlimPajama, SUGGESTED_HYPERPARAMETERS_MLM
from lamoto.tasks.mlm import MaskedLMHeadConfig, MLM_C4
from lamoto.trainer.hyperparameters import Intervals, EveryNMinutes
from lamoto.augmenting.augment_dataset import TaskWithAugmentedDataset, Truncate
from tktkt.util.environment import IS_NOT_LINUX


def makeConfig(tk: PreTrainedTokenizerBase) -> DebertaConfig:
    config = DebertaConfig.from_pretrained("microsoft/deberta-base")
    H = 512
    config.hidden_size = H
    config.intermediate_size   = H*4    # https://arxiv.org/pdf/1908.08962
    config.num_attention_heads = H//64  # https://arxiv.org/pdf/1908.08962

    config.num_hidden_layers = 6
    config.max_relative_positions = 512
    config.vocab_size = len(tk.get_vocab())
    config.tie_word_embeddings = True

    config.max_position_embeddings = 1024  # (Note: the 1024 you see in the relative positional embeddings matrix is 2*k, not this number, see docs of DebertaConfig.) The whole point of the paper is "help this tokeniser has a lot of tokens", so if we're going to truncate it better be decently far out. Can't be too far nevertheless, because pushing a full context through should still be feasible (in fact, that's what happens in packing). The use of relative embeddings means that we don't need to worry about the exact number and could arguably alter it at test time.
    return config


def deberta_pretraining(tk: PreTrainedTokenizerBase, tk_name: str, low_resource: bool):
    hp = SUGGESTED_HYPERPARAMETERS_MLM

    hp.SEED = 69420
    hp.SAVE_AS = "deberta" + "-" + tk_name + "_low"*low_resource

    # These are only to parse the config. We are going to use a HF class since LaMoTO has no weight tying yet.
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
        hp.WANDB_PROJECT = "wiat"
        hp.EXAMPLES_PER_DEVICEBATCH = 64  # Should definitely fit on an A100.
        hp.EXAMPLES_PER_EFFECTIVE_BATCH = 2048  # As in DeBERTa paper and also the best in the RoBERTa paper.

    # Training parameters
    hp.LEARNING_RATE = 5e-3  # DeBERTa uses 2e-4. I use 20x that because small learning rates are dangerous.
    hp.EFFECTIVE_BATCHES_WARMUP = 10_000
    hp.MLM_PROBABILITY = 0.15

    # Testing parameters
    hp.EXAMPLES_PER_EVALUATION = 2**14  # Two times the amount of data processed for one descent.
    hp.TRACK_BEST_MODEL = True
    hp.EVAL_VS_SAVE_INTERVALS = Intervals(
        evaluation=EveryNMinutes(minutes=60),
        checkpointing=None
    )

    ###
    if IS_NOT_LINUX:  # SlimPajama takes too long to get a stream for
        task = MLM_C4(packing=True)
    else:
        task = MLM_SlimPajama(packing=True)

    if low_resource:
        task = TaskWithAugmentedDataset(task, augmentation=Truncate(max_examples=50_000), splits={"train"})
    task.train(hp)


if __name__ == "__main__":
    from tktkt.interfaces.huggingface import TktktToHuggingFace
    from tst.experiments.tokenisers_instances import *

    if IS_NOT_LINUX:
        tk = Build_English_BPE(dropout=0.1).buildTokeniser()
        tk_name = "BPE-dropout"
        deberta_pretraining(TktktToHuggingFace(tk), tk_name, low_resource=True)
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=int)
        parser.add_argument("--low_resource", action="store_true")
        args = parser.parse_args()
        if args.model_id == 1:
            tk = Build_English_BPE(dropout=0.1).buildTokeniser()
            tk_name = "BPE-dropout"
        elif args.model_id == 2:
            tk = Build_English_Kudo(kbest=64, alpha=0.15).buildTokeniser()
            tk_name = "ULM"
        elif args.model_id in {3, 4, 5}:
            if args.model_id == 3:
                temperature = 1.0
            elif args.model_id == 4:
                temperature = +5.0
            elif args.model_id == 5:
                temperature = -10.0
            else:
                raise RuntimeError()

            tk = createTokeniser_SwitchyGrampa_BPE(
                t=temperature, l=2,
                p=0.5
            )
            tk_name = f"BPE+GRaMPa(t={temperature},l=2)"
        elif args.model_id in {6, 7, 8}:
            if args.model_id == 6:
                temperature = 1.0
            elif args.model_id == 7:
                temperature = +5.0
            elif args.model_id == 8:
                temperature = -10.0
            else:
                raise RuntimeError()

            tk = createTokeniser_SwitchyGrampa_ULM(
                kbest=1, smoothing_power=1,
                t=temperature, l=2,
                p=0.5
            )
            tk_name = f"ULM+GRaMPa(t={temperature},l=2)"
        else:
            raise ValueError("Unknown model id:", args.model_id)

        deberta_pretraining(
            TktktToHuggingFace(tk),
            tk_name=tk_name,
            low_resource=args.low_resource
        )
