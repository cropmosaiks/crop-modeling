import os
import random
from mpi4py import MPI
from task_modeling_utils import *



if __name__ == "__main__":

    directory = here("data", "random_features", "summary")
    files = [
        f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
    ]

    random.seed(42)
    n_splits = 10  # Generate n random seeds
    random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

    paramlist = get_paramlist(files, random_seeds)

    max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1
    output, oos_preds = run_simulation(paramlist, max_workers)

    save_results(output, oos_preds, n_splits)


# import os
# import pandas as pd
# import random
# from datetime import date
# from pyhere import here
# from task_modeling_utils import *
# from mpi4py.futures import MPIPoolExecutor

# directory = here("data", "random_features", "summary")
# files = [
#     f for f in os.listdir(directory) if f not in (".gitkeep", ".ipynb_checkpoints")
# ]

# random.seed(42)
# n_splits = 10  # Generate n random seeds
# random_seeds = [random.randint(0, 1_000_000) for _ in range(n_splits)]

# paramlist = [
#     (
#         f,
#         h,
#         split,
#         random_state,
#     )
#     for f in files
#     for h in [False, True]
#     for split, random_state in enumerate(random_seeds)
# ]
# paramlist = (i for i in paramlist)

# if __name__ == "__main__":
#     max_workers = int(os.environ.get("SLURM_NTASKS", 4)) - 1
#     output, oos_preds = [], []
#     executor = MPIPoolExecutor(max_workers=max_workers)
#     for out, oos in executor.starmap(model_1_sensor, paramlist):
#         output.append(out)
#         # oos_preds.append(oos)
#     executor.shutdown()

#     # Save main output
#     results = pd.DataFrame(output)
#     today = date.today().strftime("%Y-%m-%d")
#     file_name = f'1_sensor_n-splits-{n_splits}_{today}.csv'
#     print(f"Saving results as: {file_name}\n\n")
#     results.to_csv(here("data","results", file_name), index=False)

#     # Save out of sample predictions
#     # oos_results = pd.concat(oos_preds)
#     # file_name = f'1_sensor_top-mod_oos_preds_n-splits-{n_splits}_{today}.csv'
#     # print(f"Saving results as: {file_name}\n\n")
#     # oos_results.to_csv(here("data","results", file_name), index=False)
