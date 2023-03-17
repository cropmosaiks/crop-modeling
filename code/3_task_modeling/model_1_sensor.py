import os
import pandas as pd
import itertools
from datetime import date
from pyhere import here
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

directory = here("data", "random_features", "summary")
files = os.listdir(directory)
files = [f for f in files if f not in ('.gitkeep', '.ipynb_checkpoints')]
paramlist = itertools.product(files, [True, False])
paramlist = sorted(paramlist, key=lambda tup: tup[1])

# paramlist = paramlist[0:44]

paramlist = (i for i in paramlist)

if __name__ == "__main__":
    max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1

    executor = MPIPoolExecutor(max_workers=max_workers)
    output = executor.starmap(model_1_sensor, paramlist)
    executor.shutdown()

    results = pd.DataFrame(output)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'results_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    results.to_csv(here("data","results", file_name), index=False)
