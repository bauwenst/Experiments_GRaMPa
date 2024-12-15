from tst.preamble import *
from tst.experiments.deberta_hyperparameters import *

from lamoto.training.lineages import *
from lamoto.tasks import MLM_SlimPajama, DP, NER
from archit.instantiation.basemodels import DebertaBaseModel
from tktkt.util.timing import Timer
from tktkt.factories.deserialisation import BPE32ki_SlimPajama3M, KudoPiece32ki_SlimPajama3M
from tktkt.factories.tokenisers import Factory_BPE, Factory_KudoPiece, Factory_SwitchyGrampa_BPE, Factory_SwitchyGrampa_ULM

t = Timer()
print(f"=== Initialising lineages ===")
t.start(echo=True)

# Define the 8 tokenisers
bpe_vocab  = BPE32ki_SlimPajama3M()
kudo_vocab = KudoPiece32ki_SlimPajama3M()

b1,n1 = Factory_BPE(files=bpe_vocab, dropout=0.1),                               "BPE-dropout-0.1"
b2,n2 = Factory_KudoPiece(files=kudo_vocab, kbest=64, alpha=0.15),               "ULM-64-0.15"
b3,n3 = Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+1.0, l_min=2),   f"BPE+GRaMPa(t=+1.0,l=2)"
b4,n4 = Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=+5.0, l_min=2),   f"BPE+GRaMPa(t=+5.0,l=2)"
b5,n5 = Factory_SwitchyGrampa_BPE(files=bpe_vocab, p=0.5, temperature=-10.0, l_min=2),  f"BPE+GRaMPa(t=-10.0,l=2)"
b6,n6 = Factory_SwitchyGrampa_ULM(files=kudo_vocab, p=0.5, temperature=+1.0, l_min=2),  f"ULM+GRaMPa(t=+1.0,l=2)"
b7,n7 = Factory_SwitchyGrampa_ULM(files=kudo_vocab, p=0.5, temperature=+5.0, l_min=2),  f"ULM+GRaMPa(t=+5.0,l=2)"
b8,n8 = Factory_SwitchyGrampa_ULM(files=kudo_vocab, p=0.5, temperature=-10.0, l_min=2), f"ULM+GRaMPa(t=-10.0,l=2)"

# Define the 8 lineages
root1 = LineageRootNode(getConfig(b1), base_model=DebertaBaseModel, tokeniser=b1)
root2 = LineageRootNode(getConfig(b2), base_model=DebertaBaseModel, tokeniser=b2)
root3 = LineageRootNode(getConfig(b3), base_model=DebertaBaseModel, tokeniser=b3)
root4 = LineageRootNode(getConfig(b4), base_model=DebertaBaseModel, tokeniser=b4)
root5 = LineageRootNode(getConfig(b5), base_model=DebertaBaseModel, tokeniser=b5)
root6 = LineageRootNode(getConfig(b6), base_model=DebertaBaseModel, tokeniser=b6)
root7 = LineageRootNode(getConfig(b7), base_model=DebertaBaseModel, tokeniser=b7)
root8 = LineageRootNode(getConfig(b8), base_model=DebertaBaseModel, tokeniser=b8)

MODELS = LineageRegistry()
MODELS.add(Lineage(handle="1", name="deberta-" + n1, root=root1))
MODELS.add(Lineage(handle="2", name="deberta-" + n2, root=root2))
MODELS.add(Lineage(handle="3", name="deberta-" + n3, root=root3))
MODELS.add(Lineage(handle="4", name="deberta-" + n4, root=root4))
MODELS.add(Lineage(handle="5", name="deberta-" + n5, root=root5))
MODELS.add(Lineage(handle="6", name="deberta-" + n6, root=root6))
MODELS.add(Lineage(handle="7", name="deberta-" + n7, root=root7))
MODELS.add(Lineage(handle="8", name="deberta-" + n8, root=root8))

# Define pre-training
hp = getPretrainingHyperparameters()

mlm1 = root1.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-BPE-dropout_low_MLM_2024-10-15_02-33-44/checkpoint-512"))
mlm2 = root2.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-ULM_low_MLM_2024-10-15_02-40-37/checkpoint-512"))
mlm3 = root3.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-BPE+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-55/checkpoint-704"))  # !!!
mlm4 = root4.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-BPE+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-505"))
mlm5 = root5.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-BPE+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-506"))
mlm6 = root6.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-ULM+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512"))
mlm7 = root7.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-ULM+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512"))
mlm8 = root8.followUp(TrainingNode("mlm", hp=hp, trainer=TaskTrainer(), task=MLM_SlimPajama(packing=True, use_pppl=False),
                                   out="deberta-ULM+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512"))
pretraining_nodes = [mlm1, mlm2, mlm3, mlm4, mlm5, mlm6, mlm7, mlm8]

# Define fine-tuning
tuner = TaskTuner(
    warmup_steps=[50, 100, 500, 1000],
    effective_batch_sizes=[16, 32, 64, 128, 256, 512],
    learning_rates=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    adamw_decay_rates=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
)
TASK_HANDLES = []

def addFinetuningTask(handle: str, task: Task):
    hp, meta = getFinetuningHyperparameters(task)
    post_node = TuningNode(handle, hp=hp, meta=meta, tuner=tuner, task=task)
    for pre_node in pretraining_nodes:
        pre_node.followUp(post_node.duplicate())
    TASK_HANDLES.append(handle)

addFinetuningTask("pos",  POS())
addFinetuningTask("ner",  NER())
addFinetuningTask("dp",   DP())
addFinetuningTask("cola", CoLA())
addFinetuningTask("sst2", SST2())
addFinetuningTask("rte",  RTE())
addFinetuningTask("mrpc", MRPC())
addFinetuningTask("qqp",  QQP())
addFinetuningTask("qnli", QNLI())
addFinetuningTask("mnli", MNLI())
addFinetuningTask("wnli", WNLI())
addFinetuningTask("stsb", STSB())

# And now, if you want to run training from a slurm script, this is what you do:
def generateSlurmCommands(task_id: int):
    task_handle = TASK_HANDLES[task_id]
    for model in MODELS:
        print(f"sbatch deberta_run.slurm {model.handle} {task_handle} 5 512")

    # where deberta_run.slurm contains:
    #   print("python deberta_run.py --lineage $1 --task $2 --n_samples $3 --n_32batches_phase1 $4")

# print()
# for lineage in MODELS:
#     print(lineage)

print(f"=== Finished initialising lineages ===")
t.soFar(echo=True)
