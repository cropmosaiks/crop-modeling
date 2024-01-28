import os
import itertools
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
    n_splits = 1
    random.seed(42)
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    directory = here("data", "random_features", "summary")
    files = [
        f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
    ]

    anom = False
    inc_clim = False

    combinations = list(itertools.combinations(files, 2))
    kwarg_list = [
        {
            "f1": f1,
            "f2": f2,
            "he": he,
            "anomaly": anom,
            "split": split,
            "random_state": random_state,
            "include_climate": inc_clim,
            "variable_groups": None,
            "n_splits": 5,
            "return_oos_predictions": False,
        }
        for he in [True, False]
        for f1, f2 in combinations
        for split, random_state in enumerate(random_seeds)
    ]
    kwarg_list = sorted(kwarg_list, key=lambda x: x["he"])

    today = date.today().strftime("%Y-%m-%d")
    results_fn = f'2_sensor-mod_{n_splits}-splits_{today}_rcf_climate-{inc_clim}_anom-{anom}.csv'
    results_filepath = here("data", "results", results_fn)

    output = []
    futures = []
    with MPIPoolExecutor() as executor:
        for kwargs in kwarg_list:
            fn_param = kwargs.pop('f1')  
            future = executor.submit(model_2_sensor, fn_param, **kwargs)
            futures.append(future)
       
        # Open the file in append mode
        with open(results_filepath, 'a') as file:
            for future in as_completed(futures):
                result = future.result()
                # Convert the result to a DataFrame and write to CSV
                pd.DataFrame([result]).to_csv(file, header=file.tell() == 0, index=False)
                print(f"Results appended to: {results_fn}\n\n", flush=True)
                file.flush()

    print(f"Completed!")



        # for future in as_completed(futures):
        #     output.append(future.result())

    # today = date.today().strftime("%Y-%m-%d")

    # results = pd.DataFrame(output)
    # results_fn = f'2_sensor-mod_{n_splits}-splits_{today}_rcf_climate-{inc_clim}_anom-{anom}.csv'
    # print(f"Saving results as: {results_fn}\n\n")
    # results.to_csv(here("data","results", results_fn), index=False)