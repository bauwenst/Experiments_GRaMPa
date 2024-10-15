from archit.instantiation.basemodels import RobertaBaseModel

from tst.preamble import *
from tst.experiments.tokenisers_instances import getTokeniserByModelId

from typing import Set, Tuple
from transformers import PreTrainedTokenizerBase
import numpy.random as npr

from wiat.training.archit_base import DebertaBaseModel
from wiat.training.augmentation_typos import TaskWithTypos
from archit.instantiation.heads import TokenClassificationHeadConfig, SequenceClassificationHeadConfig, DependencyParsingHeadConfig, BaseModelExtendedConfig
from archit.instantiation.abstracts import HeadConfig
from lamoto.trainer.hyperparameters import getDefaultHyperparameters, TaskHyperparameters, AfterNDescents, Intervals, \
    EveryNDescentsOrOncePerEpoch, AfterNEpochs
from lamoto.tasks._core import Task, RankingMetricSpec, showWarningsAndProgress
from lamoto.tasks import *
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.util.environment import IS_NOT_LINUX
from tktkt.util.printing import dprint, pluralise


def deberta_finetuning(deberta_checkpoint: str, tokeniser: PreTrainedTokenizerBase,                         # What to test
                       task: Task, hp: TaskHyperparameters, typo_splits: Set[str], text_fields: Set[str],   # What to test on
                       n_samples: int, rank_by: RankingMetricSpec,
                       tk_name: str, task_id: int):
    MAX_EXAMPLES_PHASE1    = 8192*32  # 8192 batches at batch size 32
    EXAMPLES_BETWEEN_EVALS = 512*32   # 512 batches at batch size 32

    showWarningsAndProgress(False)
    if n_samples < 1:
        raise ValueError("At least one hyperparameter sample must be taken.")

    rng = npr.default_rng(hp.SEED + task_id + abs(hash(tk_name)))  # HPs are reproducible given the same tokeniser and task and seed.

    original_stopping_condition = hp.HARD_STOPPING_CONDITION

    if typo_splits:
        task = TaskWithTypos(task, text_fields=text_fields, splits=typo_splits, p=0.10)

    # PHASE 1: Try to find the optimal set of hyperparameters from n grid samples.
    hp.init_weights = True
    hp.MODEL_CONFIG_OR_CHECKPOINT = deberta_checkpoint
    hp.TOKENISER = tokeniser
    hp.SAVE_AS = "deberta" + "-" + tk_name
    if IS_NOT_LINUX:
        max_device_batch_size = 32
        MAX_EXAMPLES_PHASE1   = 16*32
    else:
        # hp.WANDB_PROJECT = "wiat"  # Don't log these models to WandB.
        max_device_batch_size = 128  # Even when packing 1024 tokens per example, this fits on an A100. 94% VRAM usage though. Tight.

    # Hardcoded grids for now. Probably want to make all of this customisable in the future.
    warmups               = [50, 100, 500, 1000]
    effective_batch_sizes = [16, 32, 64, 128, 256, 512]
    learning_rates        = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    decay_rates           = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    samples = zip(
        rng.choice(warmups, size=n_samples).tolist(),
        rng.choice(effective_batch_sizes, size=n_samples).tolist(),
        rng.choice(learning_rates, size=n_samples).tolist(),
        rng.choice(decay_rates, size=n_samples).tolist()
    )

    hp.EXAMPLES_PER_EVALUATION = None  # Inference should be fast enough to process anything in GLUE quickly enough.
    hp.TRACK_BEST_MODEL = True

    ranking_metric_name = "eval_" + rank_by.fullName()
    best_ranking_value  = -float("inf") if rank_by.higher_is_better else float("inf")
    argbest_hps         = None
    for tup in samples:
        wu, bs, lr, dr = tup
        hp.EFFECTIVE_BATCHES_WARMUP     = wu
        hp.EXAMPLES_PER_EFFECTIVE_BATCH = bs
        hp.learning_rate                = lr
        hp.adamw_decay_rate             = dr

        ###
        hp.HARD_STOPPING_CONDITION  = AfterNDescents(descents=int(MAX_EXAMPLES_PHASE1/bs))  # Half as many descents for double the batch size.
        hp.EVAL_VS_SAVE_INTERVALS = Intervals(
            evaluation=EveryNDescentsOrOncePerEpoch(descents=int(EXAMPLES_BETWEEN_EVALS/bs), effective_batch_size=bs),
            checkpointing=None
        )
        hp.EXAMPLES_PER_DEVICEBATCH = min(max_device_batch_size, bs)  # TODO: Should be done inside LaMoTO.
        ###

        print("\nStarting short tuning for hyperparameters:", tup)
        results = task.train(hp)
        print("Finished short tuning for hyperparameters:", tup)
        print("Results:")
        dprint(results, indent=1)
        print("="*50)

        if ranking_metric_name not in results:
            print(f"Missing ranking metric {ranking_metric_name}. Cannot rank this hyperparameter set.")
            continue

        new_ranking_value = results[ranking_metric_name]
        if rank_by.higher_is_better and new_ranking_value > best_ranking_value or \
           not rank_by.higher_is_better and new_ranking_value < best_ranking_value:
            best_ranking_value  = new_ranking_value
            argbest_hps         = tup

    if argbest_hps is None:
        raise RuntimeError(f"No hyperparameter sets resulted in the ranking metric '{ranking_metric_name}'.")
    else:
        print(f"Best hyperparameters out of {pluralise(n_samples, 'sample')} as measured by {ranking_metric_name}:", argbest_hps)

    # PHASE 2: Use the best hyperparameters you found and run until you can't.
    if IS_NOT_LINUX:
        pass
    else:
        hp.WANDB_PROJECT = "wiat"

    wu, bs, lr, dr = argbest_hps
    hp.EFFECTIVE_BATCHES_WARMUP     = wu
    hp.EXAMPLES_PER_EFFECTIVE_BATCH = bs
    hp.learning_rate                = lr
    hp.adamw_decay_rate             = dr

    ###
    hp.HARD_STOPPING_CONDITION = original_stopping_condition
    hp.EVAL_VS_SAVE_INTERVALS = Intervals(
        evaluation=EveryNDescentsOrOncePerEpoch(descents=int(EXAMPLES_BETWEEN_EVALS/bs), effective_batch_size=bs),
        checkpointing=None
    )
    hp.EXAMPLES_PER_DEVICEBATCH = min(max_device_batch_size, bs)  # TODO: Should be done inside LaMoTO.
    ###

    print("Starting long tuning for best hyperparameters:", argbest_hps)
    results = task.train(hp)
    print("Finished long tuning for best hyperparameters:", argbest_hps)
    print("Results:")
    dprint(results, indent=1)


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
        return MNLI(), s, RankingMetricSpec("f1_macro", "f1_macro", True), {"premise", "hypothesis"}
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
    hp.HARD_STOPPING_CONDITION = AfterNEpochs(epochs=10, effective_batch_size=16)  # FIXME: This is a hack. The 16 is chosen because it's the lowest batch size that can be sampled. It should really be phrased in terms of descents or minutes, because they are the unit of time and we want to limit time.

    if IS_NOT_LINUX:
        hp.archit_basemodel_class = RobertaBaseModel
        n_samples = 3
        checkpoint = "haisongzhang/roberta-tiny-cased"
        model_id  = 1
        task_id   = 1
        typo_id   = 1
        tokeniser = None
        shorthand = "test"
    else:
        hp.archit_basemodel_class = DebertaBaseModel

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_samples", type=int)
        parser.add_argument("--checkpoint", type=str)
        parser.add_argument("--old_model_id", type=int)
        parser.add_argument("--task_id", type=int)
        parser.add_argument("--typo_id", type=int)
        args = parser.parse_args()

        n_samples, checkpoint, model_id, task_id, typo_id = args.n_samples, args.checkpoint, args.old_model_id, args.task_id, args.typo_id
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
        n_samples=n_samples, rank_by=rank_by,
        tk_name=shorthand, task_id=task_id
    )
