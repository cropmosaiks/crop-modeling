import random
import pandas as pd
from pyhere import here
from datetime import date
from task_modeling_utils import *
from mpi4py.futures import MPIPoolExecutor

if __name__ == "__main__":
    # Generate n random seeds
    n_splits = 10
    random.seed(42)
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    # f1 = "landsat-c2-l2_bands-r-g-b-nir-swir16-swir22_ZMB_20k-points_1024-features_yr-2009-2021_mn-4-9_lm-True_cm-True_wa-False_summary.feather"
    # f2 = "sentinel-2-l2a_bands-2-3-4-8_ZMB_15k-points_1000-features_yr-2016-2022_mn-1-12_lm-False_cm-True_wa-False_summary.feather"

    f1 = "landsat-8-c2-l2_bands-1-2-3-4-5-6-7_ZMB_15k-points_1000-features_yr-2014-2021_mn-1-12_lm-False_cm-True_wa-False_summary.feather"
    f2 = "sentinel-2-l2a_bands-2-3-4_ZMB_4k-points_1000-features_yr-2016-2022_mn-1-12_lm-False_cm-True_wa-False_summary.feather"

    anom = True

    inc_clim = True

    # inc_clim = False
    # clim = None

    kwarg_list = [
        {
            "f1": f1,
            "f2": f2,
            "he": False,
            "anomaly": anom,
            "split": split,
            "random_state": random_state,
            "include_climate": inc_clim,
            "variable_groups": clim,
            "n_splits": 5,
            "return_oos_predictions": True,
        }
    for clim in [["ndvi"], ["ndvi", "tmp"]]
    for split, random_state in enumerate(random_seeds)
    ]

    output, oos_preds = [], []
    with MPIPoolExecutor() as executor:
        futures = [executor.submit(model_2_sensor, **kwargs) for kwargs in kwarg_list]
        for future in futures:
            out, oos = future.result()
            output.append(out)
            oos_preds.append(oos)

    today = date.today().strftime("%Y-%m-%d")

    results = pd.DataFrame(output)
    results_fn = f'2_sensor_top-mod_{n_splits}-splits_{today}_rcf_climate-{inc_clim}_anom-{anom}.csv'
    print(f"Saving results as: {results_fn}\n\n")
    results.to_csv(here("data","results", results_fn), index=False)

    oos_predictions = pd.concat(oos_preds)
    oos_fn = f'2_sensor_top-mod_oos_predictions_{n_splits}-splits_{today}_rcf_climate-{inc_clim}_anom-{anom}.csv'
    print(f"Saving results as: {oos_fn}\n\n")
    oos_predictions.to_csv(here("data","results", oos_fn), index=False)