from tst.preamble import *
from tst.experiments.tokenisers_instances import getTokeniserByModelId

from typing import Set, Tuple
from transformers import PreTrainedTokenizerBase

from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.util.environment import IS_NOT_LINUX
from archit.instantiation.heads import TokenClassificationHeadConfig, SequenceClassificationHeadConfig, DependencyParsingHeadConfig, BaseModelExtendedConfig
from archit.instantiation.abstracts import HeadConfig
from lamoto.training.auxiliary.hyperparameters import getDefaultHyperparameters, TaskHyperparameters
from lamoto.tasks._core import Task, RankingMetricSpec
from lamoto.tasks import *
from lamoto.training.core import showWarningsAndProgress, LamotoPaths
from lamoto.training.tuning import TaskTuner, MetaHyperparameters
from wiat.training.archit_base import DebertaBaseModel
from wiat.training.augmentation_typos import TaskWithTypos


def deberta_finetuning(deberta_checkpoint: str, tokeniser: PreTrainedTokenizerBase,                         # What to test
                       task: Task, hp: TaskHyperparameters, typo_splits: Set[str], text_fields: Set[str],   # What to test on
                       n_samples: int, max_batches_at_size_32: int, rank_by: RankingMetricSpec,
                       tk_name: str, task_id: int):
    showWarningsAndProgress(True)
    if n_samples < 1:
        raise ValueError("At least one hyperparameter sample must be taken.")

    if typo_splits:
        task = TaskWithTypos(task, text_fields=text_fields, splits=typo_splits, p=0.10)

    hp.MODEL_CONFIG_OR_CHECKPOINT = deberta_checkpoint
    hp.TOKENISER = tokeniser
    hp.SAVE_AS = "deberta" + "-" + tk_name
    if IS_NOT_LINUX:
        hp.EXAMPLES_PER_DEVICEBATCH = 16
    else:
        hp.WANDB_PROJECT = "wiat"
        hp.EXAMPLES_PER_DEVICEBATCH = 128  # Even when packing 1024 tokens per example, this fits on an A100. 94% VRAM usage though. Tight.

    meta = MetaHyperparameters(
        meta_seed=hp.SEED + task_id + abs(hash(tk_name)),  # HPs are reproducible given the same tokeniser and task and seed.
        n_grid_samples=n_samples,
        rank_by=rank_by,

        max_examples_phase_1=32*max_batches_at_size_32,
        max_evals_phase_1=5,  # Eval 5 times and select the version with best loss. We don't really do the evals for patience in phase 1.

        max_examples_phase_2=32*16384,
        max_evals_phase_2=32  # Eval every 512 batches.
    )

    tuner = TaskTuner(
        warmup_steps=[50, 100, 500, 1000],
        effective_batch_sizes=[16, 32, 64, 128, 256, 512],
        learning_rates=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        adamw_decay_rates=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    )
    tuner.tune(task, hp, meta)


def getTaskById(task_id: int) -> Tuple[Task, HeadConfig, RankingMetricSpec, Set[str]]:
    t = TokenClassificationHeadConfig()
    s = SequenceClassificationHeadConfig()
    d = DependencyParsingHeadConfig(
        extended_model_config=BaseModelExtendedConfig(stride=1024)
    )
    if task_id == 1:
        return CoLA(), s, RankingMetricSpec("matthews_correlation", "matthews_correlation", True), {"sentence"}
    elif task_id == 2:
        return SST2(), s, RankingMetricSpec("accuracy", "accuracy", True), {"sentence"}
    elif task_id == 3:
        return RTE(), s, RankingMetricSpec("accuracy", "accuracy", True), {"sentence1", "sentence2"}
    elif task_id == 4:
        return MRPC(), s, RankingMetricSpec("accuracy", "accuracy", True), {"sentence1", "sentence2"}  # Accuracy rather than F1 because the positives are overrepresented.
    elif task_id == 5:
        return QQP(), s, RankingMetricSpec("f1", "f1", True), {"question1", "question2"}  # Inverse reasoning here.
    elif task_id == 6:
        return QNLI(), s, RankingMetricSpec("accuracy", "accuracy", True), {"sentence1", "sentence2"}  # Accuracy rather than F1 because the dataset has a skew of exactly 50%.
    elif task_id == 7:
        return MNLI(), s, RankingMetricSpec("f1_macro", "f1", True), {"premise", "hypothesis"}
    elif task_id == 8:
        return WNLI(), s, RankingMetricSpec("accuracy", "accuracy", True), {"sentence1", "sentence2"}  # Idem
    elif task_id == 9:
        return STSB(), s, RankingMetricSpec("spearmanr", "spearmanr", True), {"sentence1", "sentence2"}
    elif task_id == 10:
        return POS(), t, RankingMetricSpec("seqeval", "overall_accuracy", True), {"tokens"}
    elif task_id == 11:
        return NER(), t, RankingMetricSpec("seqeval", "overall_f1", True), {"tokens"}
    elif task_id == 12:
        return DP(), d, RankingMetricSpec("attachment", "lcm", True), {"tokens"}
    else:
        raise ValueError("Unknown task id:", task_id)


def getTypoSplitsById(typo_id: int) -> Set[str]:
    if typo_id == 1:  # No typos
        return set()
    elif typo_id == 2:  # Only test typos
        return {"validation", "test"}
    elif typo_id == 3:  # Only train typos
        return {"train"}
    elif typo_id == 4:  # Train and test typos
        return {"train", "validation", "test"}
    else:
        return set()


if __name__ == "__main__":
    hp = getDefaultHyperparameters()
    hp.EVALS_OF_PATIENCE = 5

    if IS_NOT_LINUX:
        hp.archit_basemodel_class = DebertaBaseModel
        checkpoint = (LamotoPaths.pathToCheckpoints() / "deberta-BPE-dropout_low_MLM_2024-10-15_02-33-44" / "checkpoint-512").as_posix()
        n_samples = 3
        max_batches_at_bs32 = 128
        model_id  = 1
        task_id   = 12
        typo_id   = 1
        tokeniser, shorthand = getTokeniserByModelId(model_id)
        tokeniser = TktktToHuggingFace(tokeniser)
    else:
        hp.archit_basemodel_class = DebertaBaseModel

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_samples", type=int)
        parser.add_argument("--max_batches_at_bs32", type=int, default=1024)
        parser.add_argument("--checkpoint", type=str)
        parser.add_argument("--old_model_id", type=int)
        parser.add_argument("--task_id", type=int)
        parser.add_argument("--typo_id", type=int)
        args = parser.parse_args()

        n_samples, max_batches_at_bs32, checkpoint, model_id, task_id, typo_id = args.n_samples, args.max_batches_at_bs32, args.checkpoint, args.old_model_id, args.task_id, args.typo_id
        tokeniser, shorthand = getTokeniserByModelId(model_id)
        tokeniser = TktktToHuggingFace(tokeniser)

    if task_id == 12 and not typo_id:
        raise ValueError("Must specify a typo ID (1,2,3,4) when doing DP.")

    task, head_config, rank_by, text_fields = getTaskById(task_id)
    typo_splits                             = getTypoSplitsById(typo_id)

    hp.archit_head_config = head_config
    deberta_finetuning(
        deberta_checkpoint=checkpoint, tokeniser=tokeniser,
        task=task, hp=hp, text_fields=text_fields, typo_splits=typo_splits,
        n_samples=n_samples, max_batches_at_size_32=max_batches_at_bs32, rank_by=rank_by,
        tk_name=shorthand, task_id=task_id
    )
