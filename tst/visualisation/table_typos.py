from tst.preamble import *

from enum import Enum
from typing import Tuple, Union, TypeVar, Dict, Set

Number = Union[int,float]
T = TypeVar("T")
SortingKey = Tuple[Number,Number,Number,Number]
TableKey = Tuple[str,str,str,str]
Results = Dict[str, Dict[str, Dict[TableKey, Dict[str, float]]]]

from pathlib import Path
from csv import DictReader
import re

from fiject import Table, ColumnStyle, CacheMode
from tktkt.util.printing import rprint


###
# Input strings: these are used to parse the data.
###
GRAMPA = "GRaMPa"
BPE = "BPE"
ULM = "ULM"
BPEdropout = "BPE-dropout"

TASK = "DP"

TYPOS_NONE = None  # Not the same as "", which would be for the task string "DP+".
TYPOS_TRAINTEST = "typosLD1(train,validation,test)"
TYPOS_TRAIN = "typosLD1(train)"
TYPOS_TEST  = "typosLD1(validation,test)"

# Regexes
find_grampa_temperature = re.compile("t=(.+?),")
find_grampa_length      = re.compile("l=(.+?)\)")
find_task               = re.compile("_(.+?)(\+(.+?))?_(\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d)")

###
# Output: these are used to define the strings and orders in the table.
###
# Ordering within one row field
TOKENISERS = [BPEdropout, ULM, GRAMPA]
VOCABS     = [BPE, ULM]

# Ordering between the row fields
class KeyEntry(Enum):
    VOCAB      = 1
    TOKENISER  = 2
    CONSTRAINT = 3
    SKEW       = 4

# KEY_ORDER = [KeyEntry.TOKENISER, KeyEntry.CONSTRAINT, KeyEntry.SKEW, KeyEntry.VOCAB]
KEY_ORDER = [KeyEntry.VOCAB, KeyEntry.TOKENISER, KeyEntry.CONSTRAINT, KeyEntry.SKEW]

# Column ordering
WRAPPER_ORDER = [TYPOS_TEST, TYPOS_TRAINTEST, TYPOS_TRAIN, None]

# Output strings

# def formatTypo(yes=False, train=False) -> str:
#     return r"\checkmark" if yes else r"$\times$"
# TYPO_SEP = " / "

def formatTypo(yes: bool, train: bool) -> str:
    split = "tr" if train else "te"
    return r"\typo{" + split + "}" if yes else split
TYPO_SEP = "/"

FORMATTED_TYPOS = {
    TYPOS_NONE:      formatTypo(False,True) + TYPO_SEP + formatTypo(False,False),
    TYPOS_TRAIN:     formatTypo(True,True)  + TYPO_SEP + formatTypo(False,False),
    TYPOS_TEST:      formatTypo(False,True) + TYPO_SEP + formatTypo(True,False),
    TYPOS_TRAINTEST: formatTypo(True,True)  + TYPO_SEP + formatTypo(True,False)
}
FORMATTED_RESULTS = {
    "uas": "UAS",
    "las": "LAS",
    "ucm": "UCM",
    "lcm": "LCM"
}
TASKS_TO_METRICS_TO_RESULTS = {
    "DP": {"attachment": ["uas", "las", "ucm", "lcm"]}
}


########################################################################################################################


def orderKeyFields(tokeniser: T, constraint: T, skew: T, vocab: T) -> Tuple[T, T, T, T]:
    """
    Reorders the 4 key fields given by name into the order we want them in the table.
    """
    ordered_key = [None, None, None, None]
    ordered_key[KEY_ORDER.index(KeyEntry.TOKENISER)]  = tokeniser
    ordered_key[KEY_ORDER.index(KeyEntry.CONSTRAINT)] = constraint
    ordered_key[KEY_ORDER.index(KeyEntry.SKEW)]       = skew
    ordered_key[KEY_ORDER.index(KeyEntry.VOCAB)]      = vocab
    return tuple(ordered_key)


def unorderKeyFields(ordered_key: Tuple[T, T, T, T]) -> Tuple[T, T, T, T]:
    """
    Takes the 4 key fields as ordered in the table, and orders them in the order of the arguments to orderKeyFields().
    """
    return (
        ordered_key[KEY_ORDER.index(KeyEntry.TOKENISER)],
        ordered_key[KEY_ORDER.index(KeyEntry.CONSTRAINT)],
        ordered_key[KEY_ORDER.index(KeyEntry.SKEW)],
        ordered_key[KEY_ORDER.index(KeyEntry.VOCAB)]
    )


def stringKeyToNumberKey(row: TableKey) -> SortingKey:
    tokeniser, constraint, skew, vocab = unorderKeyFields(row)

    constraint = constraint.replace("$", "")
    if "=" in constraint:
        constraint = constraint[constraint.find("=")+1:]
    constraint = int(constraint) if constraint else 1

    skew = skew.replace("$", "")
    if "=" in skew:
        skew = skew[skew.find("=")+1:]
    skew = float(skew) if skew else 1

    return orderKeyFields(tokeniser=TOKENISERS.index(tokeniser), constraint=constraint, skew=-1/skew, vocab=VOCABS.index(vocab))


def parseWandbCsv(csv_path: Path) -> Results:
    """
    Turn Weights&Biases CSV export into standard results format.
    """
    # Gather data
    task_to_rows_to_results = dict()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = DictReader(handle)
        for row in reader:
            model_name = row["Name"]
            # First get the table key
            if GRAMPA in model_name:
                t = float(find_grampa_temperature.search(model_name).group(1))
                l = int(find_grampa_length.search(model_name).group(1))

                ### We didn't actually train such models; this is a labelling error.
                if t == 1.0 and l == 1:
                    continue
                ###

                if BPE + "+" in model_name:
                    key = orderKeyFields(tokeniser=GRAMPA, constraint=r"$\ell_\text{min}=" + f"{l}$", skew=fr"$\tau={t}$", vocab=BPE)
                elif ULM + "+" in model_name:
                    key = orderKeyFields(tokeniser=GRAMPA, constraint=r"$\ell_\text{min}=" + f"{l}$", skew=fr"$\tau={t}$", vocab=ULM)
                else:
                    print("Unparsable name:", model_name)
                    continue
            elif BPE in model_name:
                p = 0.1  # TODO: I should probably add the value of p to the pretraining shorthand, come to think of it.
                key = orderKeyFields(tokeniser=BPEdropout, constraint="", skew=fr"$p={p}$", vocab=BPE)
            elif ULM in model_name:
                k = 64
                a = 0.15
                key = orderKeyFields(tokeniser=ULM, constraint=fr"$k={k}$", skew=fr"$\alpha={a}$", vocab=ULM)
            else:
                print("Unparsable name:", model_name)
                continue
            task_name = find_task.search(model_name).group(1)
            wrapper   = find_task.search(model_name).group(3)  # Can be None and that's also valid.

            task_results = {k.replace("/", "_"): float(v) for k,v in row.items() if "test" in k and "" != v}
            if not task_results:
                continue
            assert row["State"] == "finished"

            # print("Found row", key, "for task", task_name)
            # print("\t", task_results)
            if task_name not in task_to_rows_to_results:
                task_to_rows_to_results[task_name] = dict()
            if wrapper not in task_to_rows_to_results[task_name]:
                task_to_rows_to_results[task_name][wrapper] = dict()

            if key in task_to_rows_to_results[task_name][wrapper]:
                print("\nFound duplicate key:")
                print("Task:", task_name, wrapper)
                print("Key:", key)
                print("\t Existing:", task_to_rows_to_results[task_name][wrapper][key])
                print("\tDiscarded:", task_results)
                print("\twith raw name:", model_name)
                continue

            task_to_rows_to_results[task_name][wrapper][key] = task_results

    return task_to_rows_to_results


def tableFromTasksToModelsToResults(tasks_to_wrappers_to_models_to_results: Results):
    """
    Format-independent conversion to a Fiject table.
    """
    # Define structure
    table = Table(f"finetuning-{TASK}-typos", caching=CacheMode.NONE, overwriting=True)

    rows: Set[TableKey] = set()
    for wrappers in tasks_to_wrappers_to_models_to_results.values():
        for models_to_results in wrappers.values():
            models = models_to_results.keys()
            rows.update(models)

    print("\nParsed results:")
    rprint(tasks_to_wrappers_to_models_to_results)
    if TASK not in tasks_to_wrappers_to_models_to_results:
        raise RuntimeError(f"{TASK} not in results...")

    # Sort the keys and interpret the results into columns.
    for row in sorted(rows, key=stringKeyToNumberKey):
        for wrapper in WRAPPER_ORDER:
            if wrapper not in tasks_to_wrappers_to_models_to_results[TASK]:
                continue
            if row not in tasks_to_wrappers_to_models_to_results[TASK][wrapper]:
                continue

            task_results_for_this = tasks_to_wrappers_to_models_to_results[TASK][wrapper][row]
            for metric_name, result_names in TASKS_TO_METRICS_TO_RESULTS[TASK].items():  # "attachment" -> ["uas", "las", ...]
                for result_name in result_names:
                    col = [FORMATTED_RESULTS[result_name], FORMATTED_TYPOS[wrapper]]
                    metric_with_result_key = f"test_{metric_name}_{result_name}"
                    if metric_with_result_key in task_results_for_this:
                        table.set(task_results_for_this[metric_with_result_key], list(row), col)

    table.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[0,1,2],
        default_column_style=ColumnStyle(
            do_bold_maximum=True,
            cell_prefix=r"\tgrad[0.0][0.5][1.0]{", cell_suffix="}",
            cell_default_if_empty=r"\cellcolor{black!10}")
    )


def wandbToTable(csv_path: Path):
    tableFromTasksToModelsToResults(parseWandbCsv(csv_path))


if __name__ == "__main__":
    wandbToTable(PATH_DATA_OUT / "wandb-DP.csv")
