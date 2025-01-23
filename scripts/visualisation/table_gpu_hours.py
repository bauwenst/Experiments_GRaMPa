from scripts.preamble import *

from typing import List, Callable, TypeVar, Dict, Optional
from collections import Counter
from csv import DictReader

wandb_export = PATH_DATA_OUT / "wandb-5.csv"

GPU_UNITS = {
    "A100": 1.0,
    "H100": 2.0,
}

GPU_REMAP = {
    "A100": "A100",
    "H100": "H100",
    "unknown": "A100"
}


def fuzzyIndex(l: List[str], sub: str) -> int:
    """
    Find index of the first element CONTAINING the given key.
    """
    for i,s in enumerate(l):
        if sub in s:
            return i
    else:
        return -1


T = TypeVar("T")
def reverseFuzzyGet(d: Dict[str, T], full: str, default: T=None) -> Optional[T]:
    """
    Find value of the key CONTAINED BY the given key.
    """
    best_key = ""  # Technically not fully correct, since "" could itself be a key, just like None could be.
    for k in d:
        if k in full and len(k) > len(best_key):
            best_key = k

    if best_key:
        return d[best_key]
    else:
        return default


from fiject import Table, ColumnStyle
t = Table("gpus", overwriting=True)


with open(wandb_export, "r", encoding="utf-8") as handle:
    pretraining_gpus = Counter()
    finetuning_gpus  = Counter()

    reader = DictReader(handle)
    for row in reader:
        state = row["State"]
        if state not in {"finished", "failed"}:
            print("Filtered:", row["Name"])
            continue

        runtime_s = int(row["Runtime"])

        tags = row["Tags"].split(", ")
        i = fuzzyIndex(tags, "NVIDIA")
        if i >= 0:
            gpu_tag = tags[i]
            tags.pop(i)
        else:
            gpu_tag = "unknown"

        gpu_tag = reverseFuzzyGet(GPU_REMAP, gpu_tag, gpu_tag)

        if fuzzyIndex(tags, "MLM") >= 0:
            pretraining_gpus[gpu_tag] += runtime_s
        else:
            finetuning_gpus[gpu_tag] += runtime_s

    all_gpus = pretraining_gpus + finetuning_gpus

    def printGpuForPhase(training_phase: str, summary: Counter):
        def printGpuForPhaseForTime(summary: Counter, time_name: str, seconds_transform: Callable[[float], float]):
            print(f"\n{training_phase} GPU {time_name}:")
            total = 0
            for gpu, seconds in sorted(summary.items()):
                this = seconds_transform(seconds)
                print("\t", gpu, ":", this)
                t.set(this, [training_phase], [gpu, time_name])
                total += this * reverseFuzzyGet(GPU_UNITS, gpu, 1.0)
            print("\t", "=> FLOP-adjusted sum", ":", total)
            t.set(total, [training_phase], [r"A100-eq.\ total", time_name])

        # printGpuForPhaseForTime(summary, "seconds", lambda x: x)
        # printGpuForPhaseForTime(summary, "minutes", lambda x: x/60)
        printGpuForPhaseForTime(summary, "hours", lambda x: x/60/60)
        printGpuForPhaseForTime(summary, "days", lambda x: x/60/60/24)

    printGpuForPhase("pre-training", pretraining_gpus)
    print("="*75)
    printGpuForPhase("fine-tuning", finetuning_gpus)
    print("="*75)
    printGpuForPhase("all", all_gpus)

    t.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[],
        default_column_style=ColumnStyle(
            alignment="r",
            cell_default_if_empty=r"\cellcolor{black!10}"
        )
    )
