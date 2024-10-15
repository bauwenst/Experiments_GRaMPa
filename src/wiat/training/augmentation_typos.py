from typing import Set

from lamoto.tasks._core import Task
from lamoto.augmenting.augment_dataset import MapWords, TaskWithAugmentedDataset
from tktkt.preparation.perturbers import *


class TyposLevenshtein1(MapWords):
    def __init__(self, text_field_name: str, p: float=0.10):
        sampler = ConstantSampler(n=1)
        super().__init__(
            text_field_name=text_field_name,
            mapping=ParallelPerturber(
                p=p,
                perturbations=[
                    Substitute(0, sampler=sampler),
                    Insert(0, sampler=sampler),
                    Pop(0, sampler=sampler)
                ]
            )
        )


class TaskWithTypos(TaskWithAugmentedDataset):

    def __init__(self, task: Task, text_fields: Set[str], splits: Set[str], p: float):
        text_fields = set(text_fields)
        if text_fields:
            field = text_fields.pop()
            if text_fields:
                super().__init__(TaskWithTypos(task, text_fields, splits=splits, p=p),
                                 augmentation=TyposLevenshtein1(field, p),
                                 splits=splits)
            else:
                super().__init__(task,
                                 augmentation=TyposLevenshtein1(field, p),
                                 splits=splits)
        else:
            raise ValueError("At least one text field is required to apply typos to.")
