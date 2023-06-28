import os
import random
import pandas as pd
import itertools
from pyhere import here
from datetime import date
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

if __name__ == "__main__":
    # Generate n random seeds
    n_splits = 10
    random.seed(42)
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    directory = here("data", "random_features", "summary")
    files = [
        f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
    ]
    # files = [f for f in files if not (f.startswith("landsat-8") and "lm-False" in f)]
    # files = [f for f in files if not (f.startswith("sentinel") and "lm-True" in f)]
    files = [f for f in files if "cm-True" in f]
    files = [f for f in files if "wa-False" in f]

    combinations = list(itertools.combinations(files, 2))
    combinations = [
        t for t in combinations if not ("landsat-c2" in t[0] and "landsat-c2" in t[1])
    ]

    kwarg_list = [
        {
            "f1": f1,
            "f2": f2,
            "he": True,
            "anomaly": False,
            "split": split,
            "random_state": random_state,
            "include_climate": False,
            "variable_groups": None,
            "n_splits": 5,
            "return_oos_predictions": False, 
        }
        for f1, f2 in combinations
        for split, random_state in enumerate(random_seeds)
    ]

    chunked_kwarg_list = list(chunks(kwarg_list, 20))
    j = 1
    for i, chunk in enumerate(chunked_kwarg_list):
        with MPIPoolExecutor() as executor:
            output = list(executor.map(unpack_and_run, chunk))
        results = pd.DataFrame(output)
        today = date.today().strftime("%Y-%m-%d")
        file_name = f'2_sensor_n-splits-{n_splits}_{today}_{i+j}.csv'
        print(f"Saving results as: {file_name}\n\n")
        results.to_csv(here("data","results", file_name), index=False)