from tst.preamble import *

from wiat.training.archit_base import DebertaBaseModel
from lamoto.tasks import POS, getDefaultHyperparameters
from lamoto.tasks.pos import TokenClassificationHeadConfig
from lamoto.trainer.hyperparameters import AfterNEpochs


def deberta_finetuning():
    hp = getDefaultHyperparameters()
    hp.archit_basemodel_class = DebertaBaseModel
    hp.archit_head_config = TokenClassificationHeadConfig()
    hp.MODEL_CONFIG_OR_CHECKPOINT = "microsoft/deberta-base"
    hp.HARD_STOPPING_CONDITION = AfterNEpochs(epochs=1, effective_batch_size=hp.EXAMPLES_PER_EFFECTIVE_BATCH)

    task = POS()
    task.train(hp)


if __name__ == "__main__":
    deberta_finetuning()
