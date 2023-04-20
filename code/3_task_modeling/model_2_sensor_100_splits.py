import os
import pandas as pd
import random
from datetime import date
from pyhere import here
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

random.seed(42)

# Define the number of stratified random splits to perform
n_splits = 100  # Generate n random seeds
random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

f1 = "landsat-c2-l2_bands-r-g-b-nir-swir16-swir22_ZMB_20k-points_1024-features_yr-2009-2021_mn-1-12_lm-False_cm-True_wa-False_summary.feather"
f2 = "sentinel-2-l2a_bands-2-3-4-8_ZMB_15k-points_1000-features_yr-2016-2022_mn-1-12_lm-False_cm-True_wa-False_summary.feather"

paramlist = [
    (
        f1,
        f2,
        "True",
        split,
        random_state,
    )
    for split, random_state in enumerate(random_seeds)
]

if __name__ == "__main__":
    max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1
    output, oos_preds = [], []
    executor = MPIPoolExecutor(max_workers=max_workers)
    for out, oos in executor.starmap(model_2_sensor, paramlist):
        output.append(out)
        oos_preds.append(oos)
    executor.shutdown()
    
    # Save main output
    results = pd.DataFrame(output)
    today = date.today().strftime("%Y-%m-%d")
    file_name = f'2_sensor_top-mod_n-splits-{n_splits}_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    results.to_csv(here("data","results", file_name), index=False)

    # Save out of sample predictions
    oos_results = pd.concat(oos_preds)
    file_name = f'2_sensor_top-mod_oos_preds_n-splits-{n_splits}_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    oos_results.to_csv(here("data","results", file_name), index=False)
