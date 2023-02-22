import os
import re
import pandas as pd
import itertools
import multiprocessing
from datetime import date
from pyhere import here
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

point_pattern = re.compile("20k-points")
wa_pattern = re.compile("cm-False")
directory = here("data", "random_features", "summary")
files = os.listdir(directory)
files = [f for f in files if f not in ('.gitkeep', '.ipynb_checkpoints')]
files = [f for f in files if not (bool(point_pattern.search(f)) & bool(wa_pattern.search(f)))]
paramlist = list(itertools.product(files, [True, False]))

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

