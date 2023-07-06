import os
import random
import pandas as pd
import itertools
import time
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
    files = [f for f in files if not (f.startswith("landsat-8") and "lm-False" in f)]
    files = [f for f in files if not (f.startswith("sentinel") and "lm-True" in f)]
    files = [f for f in files if "cm-True" in f]
    # files = [f for f in files if "wa-False" in f]
    
    combinations = list(itertools.combinations(files, 2))
    combinations = [
        t for t in combinations if not ("landsat-c2" in t[0] and "landsat-c2" in t[1])
    ]

    anom = False
    climate = False

    kwarg_list = [
        {
            "f1": f1,
            "f2": f2,
            "he": True,
            "anomaly": anom,
            "split": split,
            "random_state": random_state,
            "include_climate": climate,
            "variable_groups": None,
            "n_splits": 5,
            "return_oos_predictions": False, 
        }
        for f1, f2 in combinations
        for split, random_state in enumerate(random_seeds)
    ]

    chunked_kwarg_list = list(chunks(kwarg_list, 60))

    for i in range(17, len(chunked_kwarg_list)):
    # for i in range(9, 12):
        print(f"Starting iteration {i+1} of {len(chunked_kwarg_list)}\n\n")
        chunk = chunked_kwarg_list[i]
        tic = time.time()
        with MPIPoolExecutor() as executor:
            output = list(executor.map(unpack_and_run, chunk))
        toc = time.time()
        elapsed_time = toc - tic
        results = pd.DataFrame(output)
        today = date.today().strftime("%Y-%m-%d")
        file_name = f'2_sensor_{n_splits}-splits_{today}_{i+1}_rcf_climate-{climate}_anom-{anom}.csv'  
        print(f"Saving results as: {file_name}\n\n")
        results.to_csv(here("data","results", file_name), index=False)
        print(f"Elapsed time for iteration {i+1}: {elapsed_time} seconds\n\n")

