"""
Fine-tuning task for seeing if a model can learn to predict how many characters are in a token.
"""
from datasets import Dataset, DatasetDict

import evaluate

from transformers import AutoModelForTokenClassification, AutoTokenizer, RobertaTokenizerFast, TrainingArguments, Trainer
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification

import numpy as np


BATCH_SIZE = 16
TRAINING_EPOCHS = 20


def makeDataset():
    # Make dataset by looking inside the tokeniser
    tk: RobertaTokenizerFast = AutoTokenizer.from_pretrained("roberta-base")
    vocab = tk.get_vocab()
    reverse_vocab = {v:k for k,v in vocab.items()}

    skip_keys = set(tk.special_tokens_map.values())
    dataset = {id: len(typ) for typ,id in vocab.items()
                            if len(typ) < 20 and typ not in skip_keys}  # Everything past 20 is bogus.


    def tokenise(example):
        id = example["id"]
        all_tokens = tk.build_inputs_with_special_tokens([id])
        attention = [1]*len(all_tokens)
        labels    = [-100]*len(all_tokens)
        labels[all_tokens.index(id)] = example["label"]
        return {"input_ids": all_tokens, "attention_mask": attention, "labels": labels}

    dataset = Dataset.from_list([{"id": i, "label": l}
                                 for i, l in dataset.items()])
    dataset = dataset.map(tokenise)
    dataset = dataset.remove_columns(["id", "label"])

    # 80-10-10 split
    datasetdict_train_vs_validtest = dataset.train_test_split(train_size=80/100)
    datasetdict_valid_vs_test      = datasetdict_train_vs_validtest["test"].train_test_split(train_size=50/100)
    return DatasetDict({
        "train": datasetdict_train_vs_validtest["train"],
        "valid": datasetdict_valid_vs_test["train"],
        "test": datasetdict_valid_vs_test["test"]
    })


def train():
    datasetdict = makeDataset()

    model: RobertaForTokenClassification = AutoModelForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=20
    )

    # Freeze backbone https://discuss.huggingface.co/t/how-to-freeze-layers-using-trainer/4702
    for param in model.base_model.parameters():
        param.requires_grad = False

    model.to("cuda")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="../../../data/checkpoints",
        save_strategy="no",
        # load_best_model_at_end=False,

        num_train_epochs=TRAINING_EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        report_to="none",  # Disables weights-and-biases login requirement

        evaluation_strategy="epoch",
        logging_strategy="no"
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasetdict["train"],

        eval_dataset=datasetdict["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    dprint(trainer.evaluate())


def compute_metrics(predictions_labels_batch):
    """
    For some reason, seqeval doesn't want to accept arbitrary labels, only BIO format. Certified weird.
    """
    examples_tokens_classes_pred, examples_tokens_labels = predictions_labels_batch  # Predictions is of shape EXAMPLES x TOKENS x CLASSES
    examples_tokens_preds = np.argmax(examples_tokens_classes_pred, axis=2)

    categorical_metric = evaluate.load("seqeval")
    categorical_results = categorical_metric.compute(
        predictions=[  # Formatted as EXAMPLES x TOKENS matrix
            ["B-" + str(pred) for (pred, label) in zip(tokens_preds, tokens_labels) if label != -100]
             for tokens_preds, tokens_labels in zip(examples_tokens_preds, examples_tokens_labels)
        ],
        references=[
            ["B-" + str(label) for (pred, label) in zip(tokens_preds, tokens_labels) if label != -100]
             for tokens_preds, tokens_labels in zip(examples_tokens_preds, examples_tokens_labels)
        ]
    )

    regressive_metric = evaluate.load("mae")
    regressive_results = regressive_metric.compute(
        predictions=flatten([  # Goes from EXAMPLES x TOKENS to 1 x (EXAMPLES * TOKENS)
            [pred for (pred, label) in zip(tokens_preds, tokens_labels) if label != -100]
             for tokens_preds, tokens_labels in zip(examples_tokens_preds, examples_tokens_labels)
        ]),
        references=flatten([
            [label for (pred, label) in zip(tokens_preds, tokens_labels) if label != -100]
             for tokens_preds, tokens_labels in zip(examples_tokens_preds, examples_tokens_labels)
        ])
    )

    return {
        "precision": categorical_results["overall_precision"],
        "recall":    categorical_results["overall_recall"],
        "f1":        categorical_results["overall_f1"],
        "accuracy":  categorical_results["overall_accuracy"],
        "MAD":       regressive_results["mae"]
    }


def flatten(nested_list) -> list:
    result = []
    for thing in nested_list:
        if not isinstance(thing, list):
            result.append(thing)
        else:
            result.extend(flatten(thing))
    return result


def dprint(d: dict):
    print("{")
    for k,v in d.items():
        print("\t", k, ":", v)
    print("}")


if __name__ == "__main__":
    train()
    # print(flatten([[1,2,3,4], 5, 6, [[7,8], 9]]))