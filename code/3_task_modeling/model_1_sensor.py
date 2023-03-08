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
paramlist = list(itertools.product(files, [True, False]))
paramlist = sorted(paramlist, key=lambda tup: tup[1])

if __name__ == "__main__":
    output = []
    executor = MPIPoolExecutor()
    for result in executor.map(model_1_sensor, paramlist):
        output.append(result)
    executor.shutdown()
    results = pd.concat(output).reset_index(drop=True)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'results_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    results.to_csv(here("data","results", file_name), index=False)
