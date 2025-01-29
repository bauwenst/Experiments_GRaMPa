from scripts.preamble import *
from scripts.experiments.deberta_hyperparameters import *

from lamoto.training.lineages import *
from lamoto.tasks import *
from lamoto.augmenting.augment_dataset import TaskWithAugmentedDataset, Truncate, TaskWithTypos
from archit.instantiation.basemodels import DebertaBaseModel
from archit.instantiation.heads import *

from tktkt.util.timing import Timer
from tktkt.factories.deserialisation import BPE32ki_SlimPajama3M, KudoPiece32ki_SlimPajama3M
from tktkt.factories.tokenisers import Factory_BPE, Factory_KudoPiece, Factory_SwitchyGrampa_BPE, Factory_SwitchyGrampa_ULM
from tktkt.interfaces import Preprocessor, Vocab
from tktkt.models.kudopiece.vocabularisation import KudoPieceVocabulariser


class KudoPiece32ki_SlimPajama3M_Old(KudoPiece32ki_SlimPajama3M):
    def _buildVocabulary(self) -> Vocab:
        return KudoPieceVocabulariser.load(file_or_folder=self.getVocabFile(),
                                           existing_types={"<pad>": 0, "<mask>": 1, "<unk>": 2, "<s>": 3, "</s>": 4})


class KudoPiece32ki_SlimPajama3M_New(KudoPiece32ki_SlimPajama3M):
    def _buildVocabulary(self) -> Vocab:
        return KudoPieceVocabulariser.load(file_or_folder=self.getVocabFile(),
                                           existing_types=self._specials,
                                           extras_first=False)


t = Timer()
print(f"=== Initialising lineages ===")
t.start(echo=True)

# Define the 2 vocabularies. I did fumble the specials a little. A summary:
#   - The original models (identifiers 1 to 8):
#       - BPE's specials: <s>: 0, </s>: 1, <unk>: 2, <pad>: 3, <mask>: 4
#       - ULM's specials: <pad>: 0, <mask>: 1, <unk>: 2, <s>: 3, </s>: 4
#   - The new models (identifiers 9 to 14):
#       - BPE's specials: <s>: 0, </s>: 1, <unk>: 2, <pad>: 3, <mask>: 4
#       - ULM's specials: <s>: 32765, </s>: 32766, <unk>: 32767, <pad>: 32768, <mask>: 32769
bpe_vocab  = BPE32ki_SlimPajama3M()
kudo_vocab_old = KudoPiece32ki_SlimPajama3M_Old()
kudo_vocab_new = KudoPiece32ki_SlimPajama3M_New()

# Define the 14 lineages
prefix = "deberta-"
root1 = LineageRootNode(prefix + "BPE-dropout-0.1",
                        DebertaConfigFactory(), DebertaBaseModel,
                        tokeniser=Factory_BPE(files=bpe_vocab, dropout=0.1))
root2 = LineageRootNode(prefix + "ULM-64-0.15",
                        DebertaConfigFactory(), DebertaBaseModel,
                        tokeniser=Factory_KudoPiece(files=kudo_vocab_old, kbest=64, alpha=0.15))
root3 = LineageRootNode(prefix + "BPE+GRaMPa(t=+1.0,l=2)",
                        DebertaConfigFactory(), DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+1.0, l_min=2))
root4 = LineageRootNode(prefix + "BPE+GRaMPa(t=+5.0,l=2)",
                        DebertaConfigFactory(), DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+5.0, l_min=2))
root5 = LineageRootNode(prefix + "BPE+GRaMPa(t=-10.0,l=2)",
                        DebertaConfigFactory(), base_model=DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=-10.0, l_min=2))
root6 = LineageRootNode(prefix + "ULM+GRaMPa(t=+1.0,l=2)",
                        DebertaConfigFactory(), base_model=DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_old, p=0.5, temperature=+1.0, l_min=2))
root7 = LineageRootNode(prefix + "ULM+GRaMPa(t=+5.0,l=2)",
                        DebertaConfigFactory(), base_model=DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_old, p=0.5, temperature=+5.0, l_min=2))
root8 = LineageRootNode(prefix + "ULM+GRaMPa(t=-10.0,l=2)",
                        DebertaConfigFactory(), base_model=DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_old, p=0.5, temperature=-10.0, l_min=2))
root9 = LineageRootNode(prefix + "BPE+GRaMPa(t=+1.0,l=1)",
                        DebertaConfigFactory(), base_model=DebertaBaseModel,
                        tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+1.0, l_min=1))
root10 = LineageRootNode(prefix + "BPE+GRaMPa(t=+5.0,l=1)",
                         DebertaConfigFactory(), base_model=DebertaBaseModel,
                         tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+5.0, l_min=1))
root11 = LineageRootNode(prefix + "BPE+GRaMPa(t=-10.0,l=1)",
                         DebertaConfigFactory(), base_model=DebertaBaseModel,
                         tokeniser=Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=-10.0, l_min=1))
root12 = LineageRootNode(prefix + "ULM+GRaMPa(t=+1.0,l=1)",
                         DebertaConfigFactory(), base_model=DebertaBaseModel,
                         tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_new, p=0.5, temperature=+1.0, l_min=1))
root13 = LineageRootNode(prefix + "ULM+GRaMPa(t=+5.0,l=1)",
                         DebertaConfigFactory(), base_model=DebertaBaseModel,
                         tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_new, p=0.5, temperature=+5.0, l_min=1))
root14 = LineageRootNode(prefix + "ULM+GRaMPa(t=-10.0,l=1)",
                         DebertaConfigFactory(), base_model=DebertaBaseModel,
                         tokeniser=Factory_SwitchyGrampa_ULM(files=kudo_vocab_new, p=0.5, temperature=-10.0, l_min=1))

tree = LineagePlaceholderNode()

# Define pretraining.
mlm = tree.next(
    TrainingNode("mlm",
        hp=getPretrainingHyperparameters(), trainer=TaskTrainer(),
        task=TaskWithAugmentedDataset(MLM_SlimPajama(packing=True, use_pppl=False),
                                      augmentation=Truncate(max_examples=50_000), splits={"train"})
    )
)

# Define fine-tuning
# - Define heads
seq_head = SequenceClassificationHeadConfig()
tok_head = TokenClassificationHeadConfig()
dep_head = DependencyParsingHeadConfig(extended_model_config=PoolingAndStridingConfig(stride=DEBERTA_CONTEXT_LIMIT))
# - Define tuning metadata
tuner = TaskTuner(
    warmup_steps=[50, 100, 500, 1000],
    effective_batch_sizes=[16, 32, 64, 128, 256, 512],
    learning_rates=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    adamw_decay_rates=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
)
meta = MetaHyperparameters(
    meta_seed=0,  # This is copied and then altered per lineage and per task, all by the TaskTuner.
    n_grid_samples=EXPERIMENT_CONFIG.n_tuning_samples,

    max_examples_phase_1=32*EXPERIMENT_CONFIG.n_32batches_phase1,
    minmax_evals_phase_1=5,  # Eval 5 times and select the version with best loss. We don't really do the evals for patience in phase 1.

    max_examples_phase_2=32*16384,  # Run for at most 16384 batches of size 32, a 32x increase from phase 1.
    minmax_evals_phase_2=32         # Eval every 512 batches at batch size 32 (which has 16384 batches, so 16384/32 = 512).
)
# - Define hyperparameters
hp = getDefaultHyperparameters()
if IS_NOT_LINUX:
    hp.EXAMPLES_PER_DEVICEBATCH = 16
else:
    hp.WANDB_PROJECT = WANDB_PROJECT
    hp.store_in_hf_cache = True
    hp.EXAMPLES_PER_DEVICEBATCH = 128  # Even when packing 1024 tokens per example, 128 fits on an A100 with 94% VRAM usage.
mlm.next(TuningNode("pos",  hp=hp.withHeadConfig(tok_head), meta=meta, tuner=tuner, task=POS()))
mlm.next(TuningNode("ner",  hp=hp.withHeadConfig(tok_head), meta=meta, tuner=tuner, task=NER()))
mlm.next(TuningNode("sst2", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=SST2()))
mlm.next(TuningNode("qqp",  hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=QQP()))
mlm.next(TuningNode("mrpc", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=MRPC()))
mlm.next(TuningNode("rte",  hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=RTE()))
mlm.next(TuningNode("qnli", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=QNLI()))
mlm.next(TuningNode("mnli", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=MNLI()))
mlm.next(TuningNode("wnli", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=WNLI()))
mlm.next(TuningNode("cola", hp=hp.withHeadConfig(seq_head), meta=meta, tuner=tuner, task=CoLA()))

hp = hp.copy()
hp.EXAMPLES_PER_DEVICEBATCH = 64  # The problem with DP is that it has fragmentation issues in very high epochs (~12) at batch size 128.
task_dp = DP()
mlm.next(TuningNode("dp", hp=hp.withHeadConfig(dep_head), meta=meta, tuner=tuner,
                    task=task_dp))
mlm.next(TuningNode("dp-typos-1", hp=hp.withHeadConfig(dep_head), meta=meta, tuner=tuner,
                    task=TaskWithTypos(task_dp, splits={"validation", "test"}, p=TYPO_P)))
mlm.next(TuningNode("dp-typos-2", hp=hp.withHeadConfig(dep_head), meta=meta, tuner=tuner,
                    task=TaskWithTypos(task_dp, splits={"train"}, p=TYPO_P)))
mlm.next(TuningNode("dp-typos-3", hp=hp.withHeadConfig(dep_head), meta=meta, tuner=tuner,
                    task=TaskWithTypos(task_dp, splits={"train", "validation", "test"}, p=TYPO_P)))

########################################################################################################################

# Build registry and set checkpoints.
LINEAGES = tree.buildRegistry([root1, root2, root3, root4, root5, root6, root7, root8, root9, root10, root11, root12, root13, root14])

LINEAGES.get("1").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE-dropout_low_MLM_2024-10-15_02-33-44/checkpoint-512")
LINEAGES.get("2").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM_low_MLM_2024-10-15_02-40-37/checkpoint-512")
LINEAGES.get("3").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-55/checkpoint-704")  # !!!
LINEAGES.get("4").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-505")  # !!!
LINEAGES.get("5").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-506")  # !!!
LINEAGES.get("6").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512")
LINEAGES.get("7").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512")
LINEAGES.get("8").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512")
LINEAGES.get("9").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=+1.0,l=1)_MLM+trunc50K(train)/512")
LINEAGES.get("10").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=+5.0,l=1)_MLM+trunc50K(train)/512")
LINEAGES.get("11").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-BPE+GRaMPa(t=-10.0,l=1)_MLM+trunc50K(train)/512")
LINEAGES.get("12").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=+1.0,l=1)_MLM+trunc50K(train)/512")
LINEAGES.get("13").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=+5.0,l=1)_MLM+trunc50K(train)/512")
LINEAGES.get("14").out("mlm", LamotoPaths.pathToCheckpoints() / "deberta-ULM+GRaMPa(t=-10.0,l=1)_MLM+trunc50K(train)/512")

for lineage in LINEAGES:
    print(lineage)

print(f"=== Finished initialising lineages ===")
t.soFar(echo=True)


def getTokeniserFactories() -> List[Tuple[str,TokeniserFactory]]:
    factories = []
    for l in LINEAGES:
        n = l._node_tree
        assert isinstance(n, LineageRootNode)
        factories.append((l.name,n._tokeniser))
    return factories
