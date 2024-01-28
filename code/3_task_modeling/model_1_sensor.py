import os
import warnings
import random
import pandas as pd
from pyhere import here
from datetime import date
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import as_completed

warnings.filterwarnings(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    directory = here("data", "random_features", "summary")
    files = [
        f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
    ]

    random.seed(42)
    n_splits = 10
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    kwarg_list = [
        {
            "fn": f,
            "he": h,
            "split": split,
            "random_state": random_state,
            "n_splits": 5,
        }
        for f in files
        for split, random_state in enumerate(random_seeds)
        for h in [False, True]
    ]
    kwarg_list = sorted(kwarg_list, key=lambda x: x["he"])
    output = []
    with MPIPoolExecutor() as executor:
        futures = []
        for kwargs in kwarg_list:
            fn_param = kwargs.pop('fn') 
            future = executor.submit(model_1_sensor, fn_param, **kwargs)
            futures.append(future)

        for future in as_completed(futures):
            out, oos = future.result()
            output.append(out)

    today = date.today().strftime("%Y-%m-%d")

    results = pd.DataFrame(output)
    results_fn = f'1_sensor_{n_splits}-splits_{today}.csv'
    print(f"Saving results as: {results_fn}\n\n")
    results.to_csv(here("data","results", results_fn), index=False)
