import os
import random
from task_modeling_utils import *

if __name__ == "__main__":

    directory = here("data", "random_features", "summary")
    files = [
        f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
    ]

    random.seed(1991)
    n_splits = 10  # Generate n random seeds
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    paramlist = get_paramlist(files, random_seeds)

    max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1
    output, oos_preds = run_simulation(paramlist, max_workers)

    save_results(output, oos_preds, n_splits)
