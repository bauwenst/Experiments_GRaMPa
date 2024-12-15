
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", type=str)
    parser.add_argument("--task", type=str)
    # parser.add_argument("--typo_id", type=str)
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--n_32batches_phase1", type=int)
    args = parser.parse_args()

    from tst.constants import EXPERIMENT_CONFIG
    EXPERIMENT_CONFIG.n_tuning_samples   = args.n_samples
    EXPERIMENT_CONFIG.n_32batches_phase1 = args.n_32batches_phase1

    from tst.experiments.deberta_lineages import MODELS
    MODELS.get(args.lineage).run(args.task)
