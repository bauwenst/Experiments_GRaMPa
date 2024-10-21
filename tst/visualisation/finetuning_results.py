from tst.preamble import *

from typing import Tuple, Optional

from pathlib import Path
import os
import json

from fiject import Table, ColumnStyle, CacheMode


GRAMPA = "GRaMPa"
BPE = "BPE"
ULM = "ULM"
BPEdropout = "BPE-dropout"

# These are the task names as defined in lamoto.tasks.Task.task_name
COLA = "cola"
SST2 = "sst2"
RTE  = "rte"
MRPC = "mrpc"
QQP  = "qqp"
QNLI = "qnli"
MNLI = "mnli"
WNLI = "wnli"
STSB = "stsb"
POS  = "pos"
NER  = "ner"
DP   = "DP"

TOKENLEVEL_NAME = "Token-level"
SEQUENCELEVEL_NAME = "Sequence-level"

TOKENLEVEL    = [POS, NER, DP]
SEQUENCELEVEL = [SST2, QQP, MRPC, RTE, WNLI, QNLI, MNLI, COLA, STSB]

TOKENISERS = [BPEdropout, ULM, GRAMPA]
VOCABS     = [BPE, ULM]


def formatTaskName(task: str) -> str:
    if task == SST2:
        return "SST-2"
    elif task == STSB:
        return "STS-B"
    else:
        return task.upper()


TASKS_TO_METRICS_TO_RESULTS_TO_FORMATTED = {
    COLA: {"matthews_correlation": {"matthews_correlation": r"$\varphi$"}},
    SST2: {"accuracy":   {"accuracy": "Acc"}},
    RTE:  {"accuracy":   {"accuracy": "Acc"}},
    MRPC: {"accuracy":   {"accuracy": "Acc"}},
    QQP:  {"f1":         {"f1": "$F_1$"}},
    QNLI: {"accuracy":   {"accuracy": "Acc"}},
    MNLI: {"f1_macro":   {"f1_macro": "$F_1^*$"}},
    WNLI: {"accuracy":   {"accuracy": "Acc"}},
    STSB: {"spearmanr":  {"spearmanr": r"$\rho_s$"}, "pearsonr": {"pearsonr": r"$\rho_p$"}},
    POS:  {"seqeval":    {"overall_accuracy": "Acc"}},
    NER:  {"seqeval":    {"overall_f1": "$F_1$"}},
    DP:   {"attachment": {"uas": "UAS", "las": "LAS", "ucm": "UCM", "lcm": "LCM"}}
}


def rowSortKey(row: Tuple[str,str,str,str]) -> Tuple[int,int,float,int]:
    name, size, skew, vocab = row

    size = size.replace("$", "")
    if "=" in size:
        size = size[size.find("=")+1:]
    size = int(size) if size else 1

    skew = skew.replace("$", "")
    if "=" in skew:
        skew = skew[skew.find("=")+1:]
    skew = float(skew) if skew else 1

    return (TOKENISERS.index(name), size, -1/skew, VOCABS.index(vocab))


from csv import DictReader
import re

find_grampa_temperature = re.compile("t=(.+?),")
find_grampa_length      = re.compile("l=(.+?)\)")
find_task               = re.compile("_(.+?)_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d")



def resultsCsvToTable(csv_path: Path):
    # Gather data
    task_to_rows_to_results = dict()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = DictReader(handle)
        for row in reader:
            name = row["Name"]
            # First get the table key
            if GRAMPA in name:
                t = float(find_grampa_temperature.search(name).group(1))
                l = int(find_grampa_length.search(name).group(1))

                ### We didn't actually train such models; this is a labelling error.
                if t == 1.0 and l == 1:
                    continue
                ###

                if BPE + "+" in name:
                    key = (GRAMPA, r"$\ell_\text{min}=" + f"{l}$", fr"$\tau={t}$", BPE)
                elif ULM + "+" in name:
                    key = (GRAMPA, r"$\ell_\text{min}=" + f"{l}$", fr"$\tau={t}$", ULM)
                else:
                    print("Unparsable name:", name)
                    continue
            elif BPE in name:
                p = 0.1  # TODO: I should probably add the value of p to the pretraining shorthand, come to think of it.
                key = (BPEdropout, "", fr"$p={p}$", BPE)
            elif ULM in name:
                k = 64
                a = 0.15
                key = (ULM, fr"$k={k}$", fr"$\alpha={a}$", ULM)
            else:
                print("Unparsable name:", name)
                continue
            task_name = find_task.search(name).group(1)

            ### Severely undertrained
            if task_name == STSB:
                continue
            ###

            task_results = {k.replace("/", "_"): float(v) for k,v in row.items() if "test" in k and "" != v}
            if not task_results:
                continue

            assert row["Tags"] == task_name

            # print("Found row", key, "for task", task_name)
            # print("\t", task_results)
            if task_name not in task_to_rows_to_results:
                task_to_rows_to_results[task_name] = dict()

            if key in task_to_rows_to_results[task_name]:
                print("Found duplicate results:")
                print(task_name)
                print(key)
                print("\tOld:", task_to_rows_to_results[task_name][key])
                print("\tNew:", task_results)
                continue

            task_to_rows_to_results[task_name][key] = task_results

    tableFromTasksToModelsToResults(task_to_rows_to_results)


def resultsFolderToTable(folder: Path):
    # Gather data
    rows = set()
    task_to_rows_to_results = dict()

    _, subfolders, _ = next(os.walk(folder))
    for sf in subfolders:  # sf should look something like "deberta-BPE+GRaMPa(t=1.0,l=1)_cola_2024-10-15_16-49-36"
        # First get the table key
        if GRAMPA in sf:
            t = float(find_grampa_temperature.search(sf).group(1))
            l = int(find_grampa_length.search(sf).group(1))
            if BPE + "+" in sf:
                key = (GRAMPA, r"$\ell_\text{min}=" + f"{l}$", fr"$\tau={t}$", BPE)
            elif ULM + "+" in sf:
                key = (GRAMPA, r"$\ell_\text{min}=" + f"{l}$", fr"$\tau={t}$", ULM)
            else:
                print("Unparsable name:", sf)
                continue
        elif BPE in sf:
            p = 0.1  # TODO: I should probably add the value of p to the pretraining shorthand, come to think of it.
            key = (BPEdropout, "", fr"$p={p}$", BPE)
        elif ULM in sf:
            k = 64
            a = 0.15
            key = (ULM, fr"$k={k}$", fr"$\alpha={a}$", ULM)
        else:
            print("Unparsable name:", sf)
            continue

        rows.add(key)
        task_name = find_task.search(sf).group(1)

        # Then read its data
        path = None
        _, _, files = next(os.walk(folder / sf))
        for f in files:
            if ".json" in f:
                path = folder / sf / f
                break

        if path is None:
            print("Couldn't find .json in", (folder / sf).as_posix())
            continue

        with open(path, "r", encoding="utf-8") as handle:
            task_results = json.load(handle)

        print("Found row", key, "for task", task_name, "at", path.as_posix())
        print("\t", task_results)
        if task_name not in task_to_rows_to_results:
            task_to_rows_to_results[task_name] = dict()
        task_to_rows_to_results[task_name][key] = task_results

    tableFromTasksToModelsToResults(task_to_rows_to_results)


def tableFromTasksToModelsToResults(tasks_to_models_to_results: dict):
    # Define structure
    table = Table("finetuning", caching=CacheMode.NONE, overwriting=True)

    rows = set()
    for models_to_results in tasks_to_models_to_results.values():
        models = models_to_results.keys()
        rows.update(models)

    print(tasks_to_models_to_results)

    # Sort the keys and interpret the results into columns.
    for row in sorted(rows, key=rowSortKey):
        for family_name, tasks in zip([TOKENLEVEL_NAME, SEQUENCELEVEL_NAME], [TOKENLEVEL, SEQUENCELEVEL]):
            for task_name in tasks:
                if task_name not in tasks_to_models_to_results:
                    continue
                if row not in tasks_to_models_to_results[task_name]:
                    continue

                task_results_for_this = tasks_to_models_to_results[task_name][row]
                for metric_name, result_names in TASKS_TO_METRICS_TO_RESULTS_TO_FORMATTED[task_name].items():
                    for result_name, formatted in result_names.items():
                        col = [family_name, formatTaskName(task_name), formatted]
                        metric_with_result_key = f"test_{metric_name}_{result_name}"
                        if metric_with_result_key in task_results_for_this:
                            table.set(task_results_for_this[metric_with_result_key], row, col)

    table.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[0,1,2],
        default_column_style=ColumnStyle(
            do_bold_maximum=True,
            cell_prefix=r"\tgrad[0.0][0.5][1.0]{", cell_suffix="}",
            cell_default_if_empty=r"\cellcolor{black!10}")
    )


if __name__ == "__main__":
    # EVALUATIONS = LamotoPaths.pathToEvaluations()
    # resultsFolderToTable(EVALUATIONS)
    resultsCsvToTable(PATH_DATA_OUT / "wandb-1.csv")
