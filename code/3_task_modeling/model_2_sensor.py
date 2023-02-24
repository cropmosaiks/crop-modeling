import os
import re
import pandas as pd
import itertools
import multiprocessing
from datetime import date
from pyhere import here
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

i = 1

directory = here("data", "random_features", "summary")
files = os.listdir(directory)
files = list(f for f in files if f not in ('.gitkeep', '.ipynb_checkpoints'))
paramlist = list(itertools.combinations(files, 2))
paramlist = list(itertools.product(paramlist, [True, False]))
paramlist = list(tuple(merge(paramlist[i])) for i in range(len(paramlist)))

if i == 1:
    paramlist = paramlist[0:49]
elif i == 2:
    paramlist = paramlist[250:499]
elif i == 3:
    paramlist = paramlist[500:749]
elif i == 4:
    paramlist = paramlist[750:999]
elif i == 5:
    paramlist = paramlist[1000:1249]
elif i == 6:
    paramlist = paramlist[1250:1499]
elif i == 7:
    paramlist = paramlist[1500:1749]
elif i == 8:
    paramlist = paramlist[1750:1892]

if __name__ == "__main__":
    output = []
    executor = MPIPoolExecutor()
    for result in executor.map(model_2_sensor, paramlist):
        output.append(result)
    executor.shutdown()
    results = pd.concat(output).reset_index(drop=True)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'2_sensor_results_{i}_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    results.to_csv(here("data","results", file_name), index=False)
