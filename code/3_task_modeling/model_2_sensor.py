import os
import pandas as pd
import itertools
from datetime import date
from pyhere import here
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor
    
i = 5

files = os.listdir(here("data", "random_features", "summary"))
files = list(f for f in files if f not in ('.gitkeep', '.ipynb_checkpoints'))
paramlist = list(itertools.combinations(files, 2))
paramlist = list(itertools.product(paramlist, [True, False]))
paramlist = list(tuple(merge(paramlist[i])) for i in range(len(paramlist)))
paramlist = sorted(paramlist, key=lambda tup: tup[2])

if i == 1:
    paramlist = paramlist[0:249]
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

paramlist = (i for i in paramlist)

if __name__ == "__main__":

    print(f'Iteration: {i} of 8')

    max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1

    executor = MPIPoolExecutor(max_workers=max_workers)
    output = executor.starmap(model_2_sensor, paramlist)
    executor.shutdown()

    results = pd.DataFrame(output)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'2_sensor_results_{i}_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    results.to_csv(here("data","results", file_name), index=False)
