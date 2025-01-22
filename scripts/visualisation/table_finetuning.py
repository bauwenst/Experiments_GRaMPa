from collections import OrderedDict

from tktkt.util.printing import rprint

from scripts.preamble import *

from enum import Enum
from typing import Tuple, Union, TypeVar, Dict, Set

Number = Union[int,float]
T = TypeVar("T")
SortingKey = Tuple[Number,Number,Number,Number]
TableKey = Tuple[str,str,str,str]
Results = Dict[str, Dict[TableKey, Dict[str, float]]]

from pathlib import Path
import os
import json
from csv import DictReader
import re

from fiject import Table, ColumnStyle, CacheMode

###
# Input strings: these are used to parse the data.
###
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
DP   = "dp"

# Regexes
find_grampa_temperature = re.compile("t=(.+?),")
find_grampa_length      = re.compile("l=(.+?)\)")
find_task               = re.compile("_(.+?)_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d")

###
# Output strings: these are used to define the strings and orders in the table.
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
TOKENLEVEL_NAME = "Token-level"
SEQUENCELEVEL_NAME = "Sequence-level"

FAMILY_ORDER        = [TOKENLEVEL_NAME, SEQUENCELEVEL_NAME]
TOKENLEVEL_ORDER    = [POS, NER, DP]
SEQUENCELEVEL_ORDER = [SST2, QQP, MRPC, RTE, WNLI, COLA]  # QNLI, MNLI, STSB

FAMILY_TO_TASKS = {
    TOKENLEVEL_NAME: TOKENLEVEL_ORDER,
    SEQUENCELEVEL_NAME: SEQUENCELEVEL_ORDER
}

TASK_TO_METRICS_TO_SUBMETRICS = {  # For each task, defines an order of its metrics and within those an order of their submetrics.
    COLA: OrderedDict([("matthews_correlation", ["matthews_correlation"])]),
    SST2: OrderedDict([("accuracy",   ["accuracy"])]),
    RTE:  OrderedDict([("accuracy",   ["accuracy"])]),
    MRPC: OrderedDict([("accuracy",   ["accuracy"])]),
    QQP:  OrderedDict([("f1",         ["f1"])]),
    QNLI: OrderedDict([("accuracy",   ["accuracy"])]),
    MNLI: OrderedDict([("f1_macro",   ["f1_macro"])]),
    WNLI: OrderedDict([("accuracy",   ["accuracy"])]),
    # STSB: OrderedDict([("spearmanr",  ["spearmanr"]), ("pearsonr", ["pearsonr"])]),
    POS:  OrderedDict([("seqeval",    ["overall_accuracy"])]),
    NER:  OrderedDict([("seqeval",    ["overall_f1"])]),
    DP:   OrderedDict([("attachment", ["uas", "las", "ucm", "lcm"])])
}
SUBMETRIC_TO_FORMATTED = {
    "matthews_correlation": r"$\varphi$",
    "accuracy"            : "Acc",
    "overall_accuracy"    : "Acc",
    "f1"                  : "$F_1$",
    "overall_f1"          : "$F_1$",
    "f1_macro"            : "$F_1^*$",
    "spearmanr"           : r"$\rho_s$",
    "pearsonr"            : r"$\rho_p$",
    "uas"                 : "UAS",
    "las"                 : "LAS",
    "ucm"                 : "UCM",
    "lcm"                 : "LCM"
}
formatSubmetric = SUBMETRIC_TO_FORMATTED.get


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


def formatTaskName(task: str) -> str:
    if task == SST2:
        return "SST-2"
    elif task == STSB:
        return "STS-B"
    elif task == COLA:
        return "CoLA"
    elif task == POS:
        return "PoS"
    else:
        return task.upper()


def tableFromTasksToModelsToResults(tasks_to_models_to_results: Results):
    """
    Format-independent conversion to a Fiject table.
    """
    # Define structure
    table = Table("finetuning", caching=CacheMode.NONE, overwriting=True)

    rows: Set[Tuple[str,str,str,str]] = set()
    for models_to_results in tasks_to_models_to_results.values():
        models = models_to_results.keys()
        rows.update(models)

    rprint(tasks_to_models_to_results)

    # Sort the keys and interpret the results into columns.
    for row in sorted(rows, key=stringKeyToNumberKey):
        for family_name in FAMILY_ORDER:
            for task_name in FAMILY_TO_TASKS[family_name]:
                if task_name not in tasks_to_models_to_results:
                    continue
                if row not in tasks_to_models_to_results[task_name]:
                    continue

                task_results_for_this = tasks_to_models_to_results[task_name][row]
                for metric_name, submetric_names in TASK_TO_METRICS_TO_SUBMETRICS[task_name].items():
                    for submetric_name in submetric_names:
                        col = [family_name, formatTaskName(task_name), formatSubmetric(submetric_name)]
                        metric_with_result_key = f"test_{metric_name}_{submetric_name}"
                        if metric_with_result_key in task_results_for_this:
                            table.set(task_results_for_this[metric_with_result_key], list(row), col)

    table.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[0,1,2],
        default_column_style=ColumnStyle(
            do_bold_maximum=True,
            cell_prefix=r"\tgrad[0.0][0.5][1.0]{", cell_suffix="}",
            cell_default_if_empty=r"\cellcolor{black!10}"),
        # This alternate column style probably makes no sense. If you have a perfectly -1 correlation between labels and predictions, you have a perfect classifier, not a terrible classifier.
        # alternate_column_styles={(SEQUENCELEVEL_NAME,formatTaskName(COLA),TASKS_TO_METRICS_TO_RESULTS_TO_FORMATTED[COLA]["matthews_correlation"]["matthews_correlation"]): ColumnStyle(
        #     do_bold_maximum=True,
        #     cell_prefix=r"\tgrad[-1.0][0.0][1.0]{", cell_suffix="}",
        # )}
    )


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
                p = 0.1
                key = orderKeyFields(tokeniser=BPEdropout, constraint="", skew=fr"$p={p}$", vocab=BPE)
            elif ULM in model_name:
                k = 64
                a = 0.15
                key = orderKeyFields(tokeniser=ULM, constraint=fr"$k={k}$", skew=fr"$\alpha={a}$", vocab=ULM)
            else:
                print("Unparsable name:", model_name)
                continue
            task_name = find_task.search(model_name).group(1).lower()

            task_results = {k.replace("/", "_"): float(v) for k,v in row.items() if "test" in k and "" != v}
            if not task_results:
                continue

            # print("Found row", key, "for task", task_name)
            # print("\t", task_results)
            if task_name not in task_to_rows_to_results:
                task_to_rows_to_results[task_name] = dict()

            if key in task_to_rows_to_results[task_name]:
                print("Found duplicate results:")
                print(task_name)
                print(key)
                print("\t Existing:", task_to_rows_to_results[task_name][key])
                print("\tDiscarded:", task_results)
                print("\twith raw name:", model_name)
                continue

            task_to_rows_to_results[task_name][key] = task_results

    return task_to_rows_to_results


def parseLamotoFolder(folder: Path) -> Results:
    """
    Converts LaMoTO's `evaluations` folder into a results table.
    """
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

    return task_to_rows_to_results


def wandbToTable(csv_path: Path):
    tableFromTasksToModelsToResults(parseWandbCsv(csv_path))


def lamotoToTable(folder: Path):
    tableFromTasksToModelsToResults(parseLamotoFolder(folder))


from scripts.visualisation.table_abstract import *
from dataclasses import dataclass

@dataclass
class GRaMPaRowKey:
    vocab: str
    infer: str
    limit: Optional[int]
    smoothing: float

    def __hash__(self):
        return hash(self.vocab) + hash(self.infer) + hash(self.limit) + hash(self.smoothing)


@dataclass
class GRaMPaColumnKey:
    task: str    # E.g. "DP"
    metric: str  # E.g. "attachment"
    submetric: str  # E.g. "las"

    def __hash__(self):
        return hash(self.task) + hash(self.metric) + hash(self.submetric)


class GRaMPaFinetuningParser(CsvParser[GRaMPaRowKey, GRaMPaColumnKey, Tuple[int, int, int, float], Tuple[int, int, int]]):
    """
    Parses W&B CSV export for the finetuning runs.

    Row keys: [vocab, inference, limit, smoothing]
    Col keys: [level, task, metric]

    Comparing this class-based approach to the above function-based approach:
        - Pro: you can inherit all the code about row keys: extracting them, converting to sort keys, and formatting them.
        - Con: since sorting of the columns is now also handled with sort keys (rather than just by iterating over a global
               list that can be used both to check whether a given metric is desired AND where it goes), you need code for that too.
               Also, extracting and formatting used to be the same thing (which made sort keys extra difficult), whilst now
               we have separate code for formatting.
    """

    def __init__(self, tasks_to_metrics_to_submetrics: OrderedDict[str,OrderedDict[str,List[str]]], submetric_to_formatting: Dict[str,str]):
        self._tasks_to_metrics_to_submetrics = tasks_to_metrics_to_submetrics
        self._submetric_to_formatting = submetric_to_formatting

    def _extractRowKey(self, raw: Instance) -> RawRowKeys:
        model_name = raw["Name"]

        if GRAMPA in model_name:
            t = float(find_grampa_temperature.search(model_name).group(1))
            l = int(find_grampa_length.search(model_name).group(1))

            if BPE + "+" in model_name:
                return GRaMPaRowKey(BPE, GRAMPA, l, t)
            elif ULM + "+" in model_name:
                return GRaMPaRowKey(ULM, GRAMPA, l, t)
            else:
                raise RuntimeError(f"Unparsable name: {model_name}")
        elif BPE in model_name:
            p = 0.1
            return GRaMPaRowKey(BPE, BPEdropout, None, p)
        elif ULM in model_name:
            k = 64
            a = 0.15
            return GRaMPaRowKey(ULM, ULM, k, a)
        else:
            print("Unparsable name:", model_name)

    def _extractColResults(self, raw: Instance) -> Dict[RawColKeys, float]:
        model_name = raw["Name"]

        task_name = find_task.search(model_name).group(1).lower().replace("-", "").replace(".", "")
        if task_name not in self._tasks_to_metrics_to_submetrics:
            return dict()

        results: Dict[RawColKeys, float] = dict()
        for k,v in raw.items():
            if not(k.startswith("test/") and "" != v):
                continue

            k = k.removeprefix("test/")
            for i in range(len(k)):  # .split() with maxsplit=1 except you try all possible outcomes of doing so.
                if k[i] != "_":
                    continue
                metric,submetric = k[:i], k[i+1:]
                if metric in self._tasks_to_metrics_to_submetrics[task_name]:
                    break
            else:  # Metric is not desired by the user since it's not in the dictionary.
                continue

            col_key = GRaMPaColumnKey(task_name, metric, submetric)
            results[col_key] = float(v)

        return results

    def _filterColResults(self, raw: Instance, parsed_results: Dict[GRaMPaColumnKey, float]) -> Dict[RawColKeys, float]:
        return {k:v for k,v in parsed_results.items()
                if k.task in self._tasks_to_metrics_to_submetrics
                and k.metric in self._tasks_to_metrics_to_submetrics[k.task]
                and k.submetric in self._tasks_to_metrics_to_submetrics[k.task][k.metric]}

    def _to_sortkey_row(self, key: GRaMPaRowKey) -> SortableRowKeys:
        return (VOCABS.index(key.vocab), TOKENISERS.index(key.infer), key.limit if key.limit is not None else 0, -1/(key.smoothing+0.01))

    def _to_sortkey_col(self, key: GRaMPaColumnKey) -> SortableColKeys:
        if key.task in TOKENLEVEL_ORDER:
            family_index = 1
            task_index = TOKENLEVEL_ORDER.index(key.task)
        elif key.task in SEQUENCELEVEL_ORDER:
            family_index = 2
            task_index = SEQUENCELEVEL_ORDER.index(key.task)
        else:
            raise RuntimeError(f"Task doesn't seem to have a top-level family: {key.task}")

        submetric_index = 0
        for metric, submetrics in self._tasks_to_metrics_to_submetrics[key.task].items():
            if key.submetric in submetrics:
                submetric_index += submetrics.index(key.submetric)
                break
            else:
                submetric_index += len(submetrics)
        else:
            raise RuntimeError

        return (family_index, task_index, submetric_index)

    def _format_row(self, key: GRaMPaRowKey) -> FormattedKeys:
        vocab = key.vocab
        infer = key.infer

        if infer == GRAMPA:
            limit     = r"$\ell_\text{min}=" + f"{key.limit}$"
            smoothing = fr"$\tau={key.smoothing}$"
        elif infer == BPEdropout:
            limit = r"\vphantom{" + f"{key.smoothing}" + "}"  # <--- This way, fiject sees two different row values and can put a border between, whilst the human sees identical empty cells.
            smoothing = r"$\hspace*{-0.4em}" + f"p_d={key.smoothing}$"
        elif infer == ULM:
            limit     = fr"$k={key.limit}$"
            smoothing = fr"$\alpha={key.smoothing}$"
        else:
            raise RuntimeError(f"Failed to format key: {key}")

        return [vocab, infer, limit, smoothing]

    def _format_col(self, key: GRaMPaColumnKey) -> FormattedKeys:
        if key.task in TOKENLEVEL_ORDER:
            family = TOKENLEVEL_NAME
        elif key.task in SEQUENCELEVEL_ORDER:
            family = SEQUENCELEVEL_NAME
        else:
            raise RuntimeError()

        if key.task == SST2:
            task = "SST-2"
        elif key.task == STSB:
            task = "STS-B"
        elif key.task == COLA:
            task = "CoLA"
        elif key.task == POS:
            task = "PoS"
        else:
            task = key.task.upper()

        return [family, task, self._submetric_to_formatting[key.submetric]]


class GRaMPaTypoParser(GRaMPaFinetuningParser):  # The generic is not correct but eh whatever

    def _to_sortkey_col(self, key: GRaMPaColumnKey) -> SortableColKeys:
        submetric_index = self._tasks_to_metrics_to_submetrics[key.task][key.metric].index(key.submetric)

        is_train_typos = "train" in key.task
        is_test_typos  = "test"  in key.task
        typo_index = 4 if not is_train_typos and not is_test_typos else 3 if is_train_typos and not is_test_typos else 2 if is_train_typos and is_test_typos else 1

        return (submetric_index, typo_index)

    def _format_col(self, key: GRaMPaColumnKey) -> FormattedKeys:
        def formatTypo(yes: bool, train: bool) -> str:
            split = "tr" if train else "te"
            return r"\typo{" + split + "}" if yes else split

        is_train_typos = "train" in key.task
        is_test_typos  = "test"  in key.task

        typo_format = formatTypo(is_train_typos, True) + "/" + formatTypo(is_test_typos, False)
        return [self._submetric_to_formatting[key.submetric], typo_format]


########################################################################################################################


def run_v1():
    EVALUATIONS = LamotoPaths.pathToEvaluations()
    lamotoToTable(EVALUATIONS)


def run_v2():
    wandbToTable(PATH_DATA_OUT / "wandb-3.csv")


def run_v3():
    wandb_export = PATH_DATA_OUT / "wandb-5.csv"

    parser = GRaMPaFinetuningParser(TASK_TO_METRICS_TO_SUBMETRICS, SUBMETRIC_TO_FORMATTED)
    table = parser.toTable(wandb_export)
    table.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[0, 1, 2],
        default_column_style=ColumnStyle(
            do_bold_maximum=True,
            cell_prefix=r"\tgrad[0.0][0.5][1.0]{", cell_suffix="}",
            cell_default_if_empty=r"\cellcolor{black!10}"),
        # This alternate column style probably makes no sense. If you have a perfectly -1 correlation between labels and predictions, you have a perfect classifier, not a terrible classifier.
        # alternate_column_styles={(SEQUENCELEVEL_NAME,formatTaskName(COLA),TASKS_TO_METRICS_TO_RESULTS_TO_FORMATTED[COLA]["matthews_correlation"]["matthews_correlation"]): ColumnStyle(
        #     do_bold_maximum=True,
        #     cell_prefix=r"\tgrad[-1.0][0.0][1.0]{", cell_suffix="}",
        # )}
    )

    parser = GRaMPaTypoParser({
        "dp": TASK_TO_METRICS_TO_SUBMETRICS["dp"],
        "dp+typosld1(train)": TASK_TO_METRICS_TO_SUBMETRICS["dp"],
        "dp+typosld1(validation,test)": TASK_TO_METRICS_TO_SUBMETRICS["dp"],
        "dp+typosld1(train,validation,test)": TASK_TO_METRICS_TO_SUBMETRICS["dp"],
    }, SUBMETRIC_TO_FORMATTED)
    table = parser.toTable(wandb_export, stem_suffix="_typos")
    table.commit(
        borders_between_columns_of_level=[0],
        borders_between_rows_of_level=[0, 1, 2],
        default_column_style=ColumnStyle(
            do_bold_maximum=True,
            cell_prefix=r"\tgrad[0.0][0.5][1.0]{", cell_suffix="}",
            cell_default_if_empty=r"\cellcolor{black!10}"
        )
    )


if __name__ == "__main__":
    run_v3()
