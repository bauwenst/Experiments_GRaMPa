if __name__ == "__main__":
    from tktkt.util.environment import IS_NOT_LINUX

    if IS_NOT_LINUX:
        from scripts.experiments.lineages import LINEAGES

        def generateCommands(task_id: int):
            for lineage in LINEAGES:
                for i,node in enumerate(lineage):
                    if i == task_id:
                        if "mlm" in node.handle.lower():
                            print(f"sbatch run_a100.slurm {lineage.handle} {node.handle}")
                        else:
                            print(f"sbatch run_a100.slurm {lineage.handle} {node.handle} 5 512")

        generateCommands(task_id=2)
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--lineage", type=str)
        parser.add_argument("--node", type=str)
        # parser.add_argument("--typo_id", type=str)
        parser.add_argument("--n_samples", type=int)
        parser.add_argument("--n_32batches_phase1", type=int)
        args = parser.parse_args()

        from scripts.constants import EXPERIMENT_CONFIG
        EXPERIMENT_CONFIG.n_tuning_samples   = args.n_samples
        EXPERIMENT_CONFIG.n_32batches_phase1 = args.n_32batches_phase1

        from scripts.experiments.lineages import LINEAGES
        LINEAGES.get(args.lineage).run(args.node)
