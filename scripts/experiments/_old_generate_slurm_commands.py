ID_TO_CHECKPOINT = {
    1: "deberta-BPE-dropout_low_MLM_2024-10-15_02-33-44/checkpoint-512",
    2: "deberta-ULM_low_MLM_2024-10-15_02-40-37/checkpoint-512",

    3: "deberta-BPE+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-55/checkpoint-704",  # !!!
    4: "deberta-BPE+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-505",  # +-
    5: "deberta-BPE+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-506",  #+-

    6: "deberta-ULM+GRaMPa(t=1.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512",
    7: "deberta-ULM+GRaMPa(t=5.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512",
    8: "deberta-ULM+GRaMPa(t=-10.0,l=2)_low_MLM_2024-10-13_10-29-06/checkpoint-512",

    # 9: "deberta-BPE+GRaMPa(t=1.0,l=1)_low_MLM_2024-10-15_04-12-18/checkpoint-320",  # BEING TRAINED
    # 10:"deberta-ULM+GRaMPa(t=1.0,l=1)_low_MLM_2024-10-15_04-12-10/checkpoint-320"  # BEING TRAINED
}


def finetuningCalls(task_id: int, h100: bool, typo_id: int=None):
    device = "h100" if h100 else "a100"
    for id, path in ID_TO_CHECKPOINT.items():
        print(f"sbatch deberta-finetuning_{device}.slurm \"{path}\" {id} {task_id}" + (f" {typo_id}" if typo_id else ""))


if __name__ == "__main__":
    finetuningCalls(task_id=1, h100=True)
