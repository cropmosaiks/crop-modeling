import os
import re
from pyhere import here
from datetime import date

import pandas as pd

import itertools
import multiprocessing

from task_modeling_utils import *


point_pattern = re.compile("20k-points")
wa_pattern = re.compile("cm-False")

data_dir = here("data")
directory = here("data", "random_features", "summary")
files = os.listdir(directory)
files = [f for f in files if f not in ('.gitkeep', '.ipynb_checkpoints')]
files = [f for f in files if not (bool(point_pattern.search(f)) & bool(wa_pattern.search(f)))]
len(files)


paramlist = list(itertools.product(files, [True, False]))
len(paramlist)


#### No progress bar
workers = os.cpu_count()
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=workers) as pool:
        output = []
        for result in pool.imap_unordered(model_1_sensor, paramlist):
            output.append(result)
    results = pd.concat(output).reset_index(drop=True)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'results_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")           
    results.to_csv(here("data","results", file_name))
