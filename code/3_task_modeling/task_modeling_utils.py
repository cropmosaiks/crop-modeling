import gc
import sys
import time
import logging
import warnings
import traceback
import numpy as np
import pandas as pd

from pyhere import here
from datetime import date
from typing import List, Tuple, Union
from scipy.linalg import LinAlgWarning
from mpi4py.futures import MPIPoolExecutor
from glum import GeneralizedLinearRegressor as glm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold, 
    train_test_split, 
    GridSearchCV,
    cross_val_score,
    cross_val_predict, 
)
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def chunks(param_list, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(param_list), n):
        yield param_list[i : i + n]


def get_paramlist(files, random_seeds):
    return [
        {
            "fn": f,
            "he": h,
            "split": split,
            "random_state": random_state,
            "n_splits": 5,
        }
        for f in files
        for h in [False, True]
        for split, random_state in enumerate(random_seeds)
    ]


def call_model_1_sensor(kwargs):
    return model_1_sensor(**kwargs)


def run_simulation(paramlist, max_workers):
    output, oos_preds = [], []
    with MPIPoolExecutor(max_workers=max_workers) as executor:
        for out, oos in executor.map(call_model_1_sensor, paramlist):
            output.append(out)
            # oos_preds.append(oos)
    return output, oos_preds


def save_results(output, oos_preds, n_splits):
    today = date.today().strftime("%Y-%m-%d")

    # Save main output
    results = pd.DataFrame(output)
    file_name = f'1_sensor_n-splits-{n_splits}_{today}.csv'
    print(f"Saving results as: {file_name}\n\n")
    with open(here("data", "results", file_name), "w") as f:
        results.to_csv(f, index=False)


def get_mean_std_ste(df, groupby_columns, target_columns):
    """
    Group a pandas DataFrame and calculate mean, standard deviation, and standard error
    for a single column or a list of target columns.

    Args:
        df (pandas.DataFrame): The input pandas DataFrame
        groupby_columns (list): A list of columns to group by
        target_columns (str or list): A column or a list of columns to calculate the statistics for
        confidence_level (float, optional): The desired confidence level for the interval (default: 0.95)

    Returns:
        pandas.DataFrame: A summarized pandas DataFrame
    """

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    summary_dfs = []

    for target_column in target_columns:
        # Group the DataFrame
        grouped_df = df.groupby(groupby_columns)[target_column]

        # Calculate mean, standard deviation, and standard error
        mean = grouped_df.mean()
        std = grouped_df.std()
        sem = grouped_df.sem()
        sample_size = grouped_df.count()
        # Create the summarized DataFrame
        summary_df = pd.DataFrame(
            {
                "summary_var": target_column,
                "mean": mean,
                "std": std,
                "sem": sem,
                "sample_size": sample_size,
            }
        ).reset_index()

        summary_dfs.append(summary_df)

    # Concatenate the summary DataFrames
    combined_summary_df = pd.concat(summary_dfs, axis=0, ignore_index=True)

    return combined_summary_df


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


def split_fn(file_name):
    f            = file_name.split(sep="_")
    satellite    = f[0],
    bands        = f[1].replace("bands-", "")
    country_code = f[2],
    points       = f[3].replace("k-points", "")
    num_features = f[4].replace("-features", "")
    yrs          = f[5].replace("yr-", "")
    mns          = f[6].replace("mn-", "")
    limit_months = f[7].replace("lm-", "")
    crop_mask    = f[8].replace("cm-", "")
    weighted_avg = f[9].replace("wa-", "")
    
    return satellite, bands, country_code, points, yrs, mns, num_features, limit_months, crop_mask, weighted_avg


def merge(x, bases = (tuple, list)):
    for e in x:
        if type(e) in bases:
            for e in merge(e, bases):
                yield e
        else:
            yield e
            

def merge_files(file_list, file_type='csv', index_col=None):
    """
    Merges multiple files of a specified type into a single pandas dataframe.

    Parameters:
    file_list (list): A list of file paths to be merged.
    file_type (str, optional): The file type of the input files. Default is 'csv'.
    index_col (str, optional): The name of the column to use as the index for the merged dataframe. Default is None.

    Returns:
    pandas.DataFrame: A merged dataframe containing all data from the input files.

    """
    # check that file_type is supported
    if file_type not in ['csv', 'txt']:
        raise ValueError("Unsupported file type. Must be 'csv' or 'txt'.")

    # read and concatenate files in chunks to save memory
    chunks = []
    for file in file_list:
        if file_type == 'csv':
            chunk_reader = pd.read_csv(file, chunksize=1000)
        elif file_type == 'txt':
            chunk_reader = pd.read_csv(file, sep='\t', chunksize=1000)
        for chunk in chunk_reader:
            chunks.append(chunk)
    merged_data = pd.concat(chunks, ignore_index=True)

    # check for duplicate rows and remove them
    num_duplicates = merged_data.duplicated().sum()
    if num_duplicates > 0:
        merged_data.drop_duplicates(inplace=True)

    # set the index of the merged dataframe
    if index_col:
        merged_data.set_index(index_col, inplace=True)

    return merged_data


def demean_by_group(
    df, observed="log_yield", predicted="oos_prediction", group=["district", "val_fold"]
):
    """
    Demeans the observed and predicted columns of a DataFrame based on one or more grouping columns.

    Args:
        df (pandas.DataFrame): The input DataFrame to be modified.
        observed (str): The name of the column containing the observed values to be demeaned.
        predicted (str): The name of the column containing the predicted values to be demeaned.
        group (list of str): A list of column names to use for grouping the data. The function
        will calculate the mean of the observed and predicted columns for each unique combination
        of values in these columns.

    Returns:
        pandas.DataFrame: The input DataFrame with two new columns added, one for the demeaned
        observed values and one for the demeaned predicted values.
    """
    df[f"demean_{observed}"] = df[observed] - df.groupby(group)[observed].transform(
        "mean"
    )
    df[f"demean_{predicted}"] = df[predicted] - df.groupby(group)[predicted].transform(
        "mean"
    )
    return df


#########################################
#########################################
########### EXPANDING LAMBDA ############
#########################################
#########################################

def find_best_lambda_expanding_grid(
    X: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    start: int,
    end: int,
    penalties: List[float],
    kfold: KFold,
) -> Tuple[float, float]:

    best_score = -np.inf
    best_lambda = 0
    edge_found = False
    score_dict = {}

    while not edge_found:
        for pen in grid:
            # Skip if this penalty has been tried before
            if pen in score_dict:
                continue

            penalties[start:end] = [pen] * (end - start)

            ridge = glm(family="normal", P2=penalties, l1_ratio=0)
            score = np.mean(cross_val_score(ridge, X, y, scoring="r2", cv=kfold))

            score_dict[pen] = score

            if score > best_score:
                best_score = score
                best_lambda = pen

        # Expand the search space if an edge case is detected and within limits
        if best_lambda == grid[0] and best_lambda / 10 >= 1e-9:
            grid = np.insert(grid, 0, best_lambda / 10)
        elif best_lambda == grid[-1] and best_lambda * 10 <= 1e9:
            grid = np.append(grid, best_lambda * 10)
        else:
            edge_found = True

    # print(f"\n{score_dict.keys()}\nBest \u03BB: {best_lambda}\nBest score: {best_score}")

    return best_lambda, best_score


#########################################
#########################################
####### RR MULTI LAMBDA TUNING ##########
#########################################
#########################################


def kfold_rr_multi_lambda_tuning(
    X: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray = np.logspace(-1, 1, base=10, num=3),
    n_splits: int = 5,
    start: Union[int, List[int]] = 0,
    end: Union[int, List[int]] = 12,
    static_lam: float = 1,
    verbose: int = 0,
    show_linalg_warning: bool = False,
    fit_model_after_tuning: bool = True,
) -> Tuple[List[float], List[float], Union[glm, float]]:
    """
    Performs k-fold cross-validated ridge regression while tuning multiple
    penalization parameters for groups of features. Each group of features
    can have a unique penalization parameter.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    y : ndarray of shape (n_samples,)
        The target values.
    grid : ndarray of shape (n_lambdas,), optional (default=np.logspace(-8, 8, base=10, num=17))
        The range of lambda values to search over.
    n_splits : int, optional (default=5)
        The number of folds to use for cross-validation.
    start : int or list of ints, optional (default=0)
        The starting indices of groups of features that share a lambda.
        If an int is provided, it is assumed that all features share the same lambda.
    end : int or list of ints, optional (default=12)
        The ending indices of groups of features that share a lambda.
        If an int is provided, it is assumed that all features share the same lambda.
    static_lam : float, optional (default=1)
        The default lambda value to use for features not in any group.
    verbose : int, optional (default=0)
        Verbosity level. If > 0, prints out the best lambda value and validation R^2 for each group of features.
        If > 1, also prints out the lambda value being currently searched over.
    show_linalg_warning : bool, optional (default=False)
        Whether to show warnings related to linear algebra operations.
    fit_model_after_tuning : bool, optional (default=True)
        Whether to fit the final model using the selected lambdas.

    Returns
    -------
    lambdas : list of floats
        The selected penalization parameters for each group of features.
    best_scores : list of floats
        The validation R^2 scores achieved with the selected penalization values for each group of features.
    model : GeneralizedLinearRegressor or np.nan
        The fitted model with the selected penalization values, or np.nan if `fit_model_after_tuning` is False.
    """
    # Ignore linear algebra warnings if show_linalg_warning is False
    if not show_linalg_warning:
        warnings.filterwarnings(action="ignore", category=LinAlgWarning, module="glum")

    # Convert start and end to lists if they are not iterable
    if not hasattr(start, "__iter__"):
        start = [start]
        end = [end]

    # Ensure that start and end lists have the same length
    assert len(start) == len(end), "Start and end indexes must have the same length"

    # Create the KFold cross-validator
    kfold = KFold(n_splits=n_splits)

    # Initialize penalties with the default static_lam value
    penalties = [static_lam] * X.shape[1]
    # Initialize lists for storing best lambdas and their corresponding best_scores
    lambdas = []
    best_scores = []

    # Loop through the groups of features
    for s, e in zip(start, end):
        scores = []

    # Loop through the groups of features
    for s, e in zip(start, end):
        # Call the find_best_lambda function instead of the inner loop
        best_lambda, best_score = find_best_lambda_expanding_grid(
            X, y, grid, s, e, penalties, kfold
        )

        penalties[s:e] = [best_lambda] * (e - s)

        # Print the best lambda and validation R^2 if verbosity level is > 0
        if verbose > 0:
            print(f"""\n\tBest \u03BB: {best_lambda}\n\tVal R2: {best_score:0.4f}\n""")
            sys.stdout.flush()

        # Append the best lambda and its score to the respective lists
        lambdas.append(best_lambda)
        best_scores.append(best_score)

    # Fit the final model using the selected lambdas if fit_model_after_tuning is True
    if fit_model_after_tuning:
        for s, e, lam in zip(start, end, lambdas):
            penalties[s:e] = [lam] * (e - s)

        ridge = glm(family="normal", P2=penalties, l1_ratio=0)
        model = ridge.fit(X, y)
    else:
        model = np.nan

    return lambdas, best_scores, model


#########################################
#########################################
########### CLIMATE MODEL ###############
#########################################
#########################################

def climate_model(
    variable_groups=["pre", "tmp", "ndvi"],
    hot_encode=True,
    anomaly=False,
    index_cols=["year", "district", "yield_mt"],
    year_start=2016,
    n_splits=5,
    split=0,
    random_state=42,
    return_oos_predictions=True,
):
    variable_groups_str = "_".join(variable_groups)

    #########################################     READ DATA    #########################################
    data = pd.read_csv(here("data", "climate", "climate_summary.csv"))
    data = data.dropna()

    #########################################     FILTER DATA    #########################################
    keep_cols = []

    for var in variable_groups:
        tmp = data.columns[data.columns.to_series().str.contains(var)].tolist()
        keep_cols.append(tmp)

    keep_cols = [*index_cols, *[col for cols in keep_cols for col in cols]]
    data = data.loc[:, keep_cols]
    data = data[data.year >= year_start]

    data["log_yield"] = np.log10(data["yield_mt"] + 1)

    data["demean_log_yield"] = data.log_yield - data.groupby(
        "district"
    ).log_yield.transform("mean")

    index_cols.append("log_yield")
    index_cols.append("demean_log_yield")

    #########################################    MAKE A COPY    #########################################
    crop_yield = data.copy().loc[:, tuple(index_cols)].reset_index(drop=True)

    #########################################     CALCULATE ANOMALY   #########################################
    if anomaly:
        data.set_index(["year", "district"], inplace=True)
        var_cols = data.columns
        data = data[var_cols] - data.groupby(["district"], as_index=True)[
            var_cols
        ].transform("mean")
    else:
        pass

    data.reset_index(drop=False, inplace=True)

    #########################################    HOT ENCODE    #########################################
    if hot_encode:
        index_cols.remove("district")
        data = pd.get_dummies(data, columns=["district"], drop_first=False)
    else:
        pass

    #########################################     K-FOLD SPLIT    #########################################
    x_all = data.drop(index_cols, axis=1)
    if anomaly:
        y_all = data.demean_log_yield
    else:
        y_all = data.log_yield
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=random_state
    )
    kfold = KFold(n_splits=n_splits)

    #########################################    STANDARDIZE FEATURES    #########################################
    scaler = StandardScaler().fit(x_all)
    x_train = pd.DataFrame(
        scaler.transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )

    #########################################     K-FOLD CV    #########################################
    ### SETUP
    tic = time.time()
    alphas = {"alpha": np.logspace(-1, 1, base=10, num=3)}

    ### LAMBDA INDICIES
    i = 0
    start = [i]
    end = [x_train.shape[1]]

    for var in variable_groups:
        i += 12
        start.append(i)
        end.append(i)
    start.sort()
    end.sort()

    if not hot_encode:
        start = start[0:-1]
        end = end[0:-1]

    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER(S)
    best_lambdas, best_scores, best_model = kfold_rr_multi_lambda_tuning(
        X=x_train,
        y=y_train,
        grid=alphas.get("alpha"),
        n_splits=n_splits,
        start=start,
        end=end,
        static_lam=1,
        verbose=0,
        show_linalg_warning=False,
        fit_model_after_tuning=True,
    )
    ### PREDICT WITH BEST HYPERPARAMETER(S)
    val_predictions = cross_val_predict(best_model, X=x_train, y=y_train, cv=kfold)
    train_predictions = best_model.predict(x_train)
    test_predictions = best_model.predict(x_test)

    if anomaly:
        pass
    else:
        val_predictions = np.maximum(val_predictions, 0)
        train_predictions = np.maximum(train_predictions, 0)
        test_predictions = np.maximum(test_predictions, 0)

    #########################################     DE-MEAN TRAIN R2    #########################################
    fold_list = []
    for i in range(n_splits):
        idx = len(list(kfold.split(y_train))[i][1])
        fold = np.repeat(i + 1, idx).tolist()
        fold_list.append(fold)
    fold_list = [item for sublist in fold_list for item in sublist]

    train_split = pd.DataFrame(
        np.repeat("train", len(x_train)), columns=["data_fold"], index=x_train.index
    )
    train_split = train_split.join(crop_yield.copy()[crop_yield.index.isin(x_train.index)])
    train_split["oos_prediction"] = val_predictions
    train_split["val_fold"] = fold_list

    #########################################     DE-MEAN TEST R2    #########################################
    test_split = pd.DataFrame(
        {"data_fold": np.repeat("test", len(x_test))}, index=x_test.index
    )
    test_split = test_split.join(crop_yield.copy()[crop_yield.index.isin(x_test.index)])
    test_split["oos_prediction"] = test_predictions
    test_split["val_fold"] = n_splits + 1

    #########################################     OUT OF SAMPLE PREDICTIONS    #########################################
    oos_preds = pd.concat([train_split, test_split])
    oos_preds[["split", "random_state"]] = split, random_state
    oos_preds["variables"] = variable_groups_str
    oos_preds["anomaly"] = anomaly
    oos_preds["hot_encode"] = hot_encode
    oos_preds["year_start"] = year_start
    oos_preds["demean_oos_prediction"] = oos_preds.oos_prediction - oos_preds.groupby(
        "district"
    ).oos_prediction.transform("mean")

    #########################################     SCORES    #########################################
    val_R2 = r2_score(y_train, val_predictions)
    val_r = pearsonr(val_predictions, y_train)[0]
    train_R2 = r2_score(y_train, train_predictions)
    train_r = pearsonr(train_predictions, y_train)[0]
    test_R2 = r2_score(y_test, test_predictions)
    test_r = pearsonr(test_predictions, y_test)[0]

    if anomaly:
        demean_cv_R2 = np.nan
        demean_cv_r = np.nan
        demean_test_R2 = np.nan
        demean_test_r = np.nan
    else:
        test = oos_preds[oos_preds.data_fold == "test"]
        train = oos_preds[oos_preds.data_fold == "train"]
        demean_cv_R2 = r2_score(train.demean_log_yield, train.demean_oos_prediction)
        demean_cv_r = pearsonr(train.demean_log_yield, train.demean_oos_prediction)[0]
        demean_test_R2 = r2_score(test.demean_log_yield, test.demean_oos_prediction)
        demean_test_r = pearsonr(test.demean_log_yield, test.demean_oos_prediction)[0]

    d = {
        "split": split,
        "random_state": random_state,
        "variables": "_".join(variable_groups),
        "year_start": year_start,
        "hot_encode": hot_encode,
        "anomaly": anomaly,
        "total_n": len(x_all),
        "train_n": len(x_train),
        "test_n": len(x_test),
        "best_reg_param": best_lambdas,
        "mean_of_val_R2": best_scores,
        "val_R2": val_R2,
        "val_r": val_r,
        "val_r2": val_r**2,
        "train_R2": train_R2,
        "train_r": train_r,
        "train_r2": train_r**2,
        "test_R2": test_R2,
        "test_r": test_r,
        "test_r2": test_r**2,
        "demean_cv_R2": demean_cv_R2,
        "demean_cv_r": demean_cv_r,
        "demean_cv_r2": demean_cv_r**2,
        "demean_test_R2": demean_test_R2,
        "demean_test_r": demean_test_r,
        "demean_test_r2": demean_test_r**2,
    }
    if return_oos_predictions:
        return d, oos_preds
    else:
        return d

    
#########################################
#########################################
########### ONE SENSOR MODEL ############
#########################################
#########################################

def model_1_sensor(fn, he, split=0, random_state=42, n_splits=5):
#########################################     SET PARAMS    #########################################
    drop_cols  = ['district', 'year', 'yield_mt']
    satellite, bands, country_code, points, yrs, mns,\
    n_features, limit_months, crop_mask, weighted_avg = split_fn(fn)

    print(f"""
Begin with paramters:
    File: {fn}
    One-hot encoding: {he}
    Split: {split}
    Random_state: {random_state}
    """, flush=True)

#########################################     READ, CLEAN, AND COPY   #########################################
    features = pd.read_feather(here('data', 'random_features', 'summary', fn))
    features.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)
    
    n_districts = len(features.district.unique())

    crop_yield = features.copy().loc[:, tuple(drop_cols)]
    crop_yield["log_yield"] = np.log10(crop_yield.yield_mt.to_numpy() + 1)

########################################     HOT ENCODE    ###########################################
    if he:
        drop_cols.remove("district")
        features = pd.get_dummies(features, columns=["district"], drop_first=False)
    else:
        pass

#########################################     K-FOLD SPLIT    #########################################
    x_all = features.drop(drop_cols, axis = 1) 
    y_all = np.log10(features.yield_mt.to_numpy() + 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=random_state
        )

#########################################     K-FOLD CV    ###########################################
    ### SETUP
    ridge  = Ridge()  
    kfold  = KFold(n_splits=n_splits)
    tic = time.time()
    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER(S)
    if he:
        alphas = {'alpha': np.logspace(-1, 1, base = 10, num = 3)}
        best_lambdas, best_scores, best_model = kfold_rr_multi_lambda_tuning(
            X=x_train,
            y=y_train, 
            grid=alphas.get('alpha'), 
            n_splits=n_splits,
            start=[0, x_train.shape[1]-n_districts],
            end=[x_train.shape[1]-72, x_train.shape[1]], 
            static_lam=1,
            verbose=1,
            show_linalg_warning=False,
            fit_model_after_tuning=True
        )
    else:
        alphas = {'alpha': np.logspace(-8, 8, base = 10, num = 17)}
        search = GridSearchCV(ridge, alphas, scoring = 'r2', cv = kfold).fit(x_train, y_train)
        best_model   = search.best_estimator_
        best_scores  = search.best_score_
        best_lambdas = best_model.alpha
    ### PREDICT WITH BEST HYPERPARAMETER(S)
    val_predictions   = cross_val_predict(best_model, X=x_train, y=y_train, cv=kfold)   
    train_predictions = best_model.predict(x_train)
    test_predictions  = best_model.predict(x_test)
    print(f"""
Finish:
    File: {fn}
    One-hot encoding: {he}
    Split: {split}
    Random_state: {random_state}
    Final Val R2:  {r2_score(y_train, val_predictions):0.4f} 
    Final Test R2: {r2_score(y_test, test_predictions):0.4f}
    Total time: {(time.time()-tic)/60:0.2f} minutes
    """, flush=True)
    
    #########################################     DE-MEAN TRAIN R2    #########################################
    train_split = pd.DataFrame(
        np.repeat("train", len(x_train)), columns=["data_fold"], index=x_train.index
    )
    train_split = train_split.join(
        crop_yield.copy()[crop_yield.index.isin(x_train.index)]
    )
    train_split["oos_prediction"] = np.maximum(val_predictions, 0)
    train_split = demean_by_group(train_split, predicted="oos_prediction", group=["district"])

    #########################################     DE-MEAN TEST R2    #########################################
    test_split = pd.DataFrame({"data_fold": np.repeat("test", len(x_test))}, index=x_test.index)
    test_split = test_split.join(crop_yield.copy()[crop_yield.index.isin(x_test.index)])
    test_split["oos_prediction"] = np.maximum(best_model.predict(x_test), 0)
    test_split = demean_by_group(test_split, predicted="oos_prediction", group=["district"])

    #########################################     OUT OF SAMPLE PREDICTIONS    #########################################
    oos_preds = pd.concat([train_split, test_split])
    oos_preds[["split", "random_state"]] = split, random_state

#########################################     SAVE RESULTS    #########################################
    d = {
        'split': split,
        'random_state': random_state,
        'country'     : country_code[0],
        'satellite'   : satellite[0],
        'bands'       : bands,
        'num_features': n_features,
        'points'      : points, 
        'month_range' : mns,
        'year_range'  : yrs,

        'limit_months': limit_months,
        'crop_mask'   : crop_mask,
        'weighted_avg': weighted_avg,
        'hot_encode': he,

        'total_n': len(x_all),
        'train_n': len(x_train),
        'test_n' : len(x_test),

        'best_reg_param': [best_lambdas],
        'mean_of_val_R2': [best_scores],
        'val_R2': r2_score(y_train, val_predictions),
        'val_r' : pearsonr(val_predictions, y_train)[0],
        'val_r2': pearsonr(val_predictions, y_train)[0] ** 2,

        'train_R2': r2_score(y_train, train_predictions),
        'train_r' : pearsonr(train_predictions, y_train)[0],
        'train_r2': pearsonr(train_predictions, y_train)[0] ** 2,

        'test_R2': r2_score(y_test, test_predictions),
        'test_r' : pearsonr(test_predictions, y_test)[0],
        'test_r2': pearsonr(test_predictions, y_test)[0] ** 2,

        "demean_cv_R2": r2_score(
            train_split.demean_log_yield, train_split.demean_oos_prediction
        ),
        "demean_cv_r": pearsonr(
            train_split.demean_log_yield, train_split.demean_oos_prediction
        )[0],
        "demean_cv_r2": pearsonr(
            train_split.demean_log_yield, train_split.demean_oos_prediction
        )[0]
        ** 2,
        "demean_test_R2": r2_score(
            test_split.demean_log_yield, test_split.demean_oos_prediction
        ),
        "demean_test_r": pearsonr(
            test_split.demean_log_yield, test_split.demean_oos_prediction
        )[0],
        "demean_test_r2": pearsonr(
            test_split.demean_log_yield, test_split.demean_oos_prediction
        )[0]
        ** 2,
    }
    return d, oos_preds


#########################################
#########################################
########### TWO SENSOR MODEL ############
#########################################
#########################################

def model_2_sensor(
    f1, 
    f2, 
    he,
    anomaly=False,
    split=0, 
    random_state=42, 
    include_climate=False, 
    variable_groups = None, 
    n_splits=5,
    return_oos_predictions = False,
    ):
    #########################################     SET PARAMS    #########################################    
    satellite1, bands1, country_code, points1, yrs1, mns1,\
    num_features1, limit_months1, crop_mask1, weighted_avg1 = split_fn(f1)

    satellite2, bands2, country_code, points2, yrs2, mns2,\
    num_features2, limit_months2, crop_mask2, weighted_avg2 = split_fn(f2)

    if variable_groups is None:
        variable_groups_str = "rcf"
    else:
        variable_groups_str = "_".join(variable_groups)
        variable_groups_str = "rcf_" + variable_groups_str

    print(f"""
Begin with paramters:
    F1: {f1}
    F2: {f2}
    One-hot encoding: {he}
    Anomaly: {anomaly}
    Split: {split}
    Random state: {random_state}
    N-splits: {n_splits}
    Include climate: {include_climate}
    Climate vars: {variable_groups_str}
    """, flush=True)

    #########################################     READ DATA    #########################################
    features_1 = pd.read_feather(here('data', 'random_features', 'summary', f1))
    features_2 = pd.read_feather(here('data', 'random_features', 'summary', f2))
    if include_climate:
        climate_df = pd.read_csv(here('data', 'climate', 'climate_summary.csv'))

    #########################################     CLEAN DATA    #########################################  
    min_year = max(min(features_1.year), min(features_2.year))
    max_year = min(max(features_1.year), max(features_2.year))

    features_1 = features_1[features_1.year >= min_year]
    features_2 = features_2[features_2.year >= min_year]

    features_1 = features_1[features_1.year <= max_year]
    features_2 = features_2[features_2.year <= max_year]

    features_1.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)
    features_2.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)

    #########################################     JOIN FEATURES    #########################################  
    drop_cols = ['district', 'year', 'yield_mt']

    features_1 = features_1.set_index(drop_cols).add_prefix("f1_")
    features_2 = features_2.set_index(drop_cols).add_prefix("f2_")

    features = features_1.join(features_2).reset_index()
    features = features[~features.isna().any(axis = 1)]

    features['log_yield'] = np.log10(features['yield_mt'] + 1)

    #########################################    JOIN CLIMATE VARS    #########################################
    if include_climate:
        keep_cols = []

        for var in variable_groups:
            tmp = climate_df.columns[climate_df.columns.to_series().str.contains(var)].tolist()
            keep_cols.append(tmp)

        keep_cols = [*drop_cols, *[col for cols in keep_cols for col in cols]]

        climate_df = climate_df.loc[:, keep_cols]

        features = (
            features.set_index(drop_cols).join(climate_df.set_index(drop_cols)).reset_index()
        )
        features = features[features.year <= max(climate_df.year)]

    drop_cols.append('log_yield')

    #########################################     CALCULATE ANOMALY   #########################################
    if anomaly:
        features.set_index(['year', 'district'], inplace=True)
        var_cols = features.columns
        features = features[var_cols] - features.groupby(['district'], as_index=True)[var_cols].transform('mean')
        features.reset_index(drop=False, inplace=True)
    else:
        pass

    #########################################     CLEAN AND COPY    #########################################
    yrs = f"{min(features.year)}-{max(features.year)}"
    n_fts_1 = features_1.shape[1]
    n_fts_2 = features_2.shape[1]
    n_districts = len(features.district.unique())

    if include_climate:
        n_climate_cols = climate_df.shape[1] - len(drop_cols)

        i = 0
        n_climate_groups = []
        for cols in range(n_climate_cols):
            if cols % 12 == 0:
                i += 1
                n_climate_groups.append(i)
        n_climate_groups

    crop_yield = features.copy().loc[:, tuple(drop_cols)]

    del features_1, features_2
    gc.collect()

    #########################################    HOT ENCODE    #########################################
    if he:
        drop_cols.remove("district")
        features = pd.get_dummies(
            features, columns=["district"], drop_first=False, dtype=float
        )
    else:
        pass

    #########################################     TRAIN/TEST SPLIT    #########################################
    x_all = features.drop(drop_cols, axis=1)
    y_all = features.log_yield
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=random_state
    )

    del features
    gc.collect()

    #########################################    STANDARDIZE FEATURES    #########################################
    scaler = StandardScaler().fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    #########################################     K-FOLD CV    ###########################################
    ### SETUP
    tic = time.time()
    kfold = KFold(n_splits=n_splits)
    alphas = {"alpha": np.logspace(-1, 1, base=10, num=3)}

    ### LAMBDA INDICIES
    start = [0, n_fts_1]
    end = [n_fts_1, x_train.shape[1]]

    if include_climate:
        start.append(n_fts_1 + n_fts_2)  
        end.append(n_fts_1 + n_fts_2)  

        for n in n_climate_groups:
            x = n * 12
            y = n_fts_1 + n_fts_2 + x
            start.append(y)
            end.append(y)

    if not include_climate and he:
        start.append(x_train.shape[1] - n_districts)
        end.append(x_train.shape[1] - n_districts)

    end.sort()

    print(f'Group indicies {start}\n\t\t  {end}', end='\n\n')

    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER(S)
    best_lambdas, best_scores, best_model = kfold_rr_multi_lambda_tuning(
        X=x_train,
        y=y_train, 
        grid=alphas.get('alpha'), 
        n_splits=n_splits,
        start=start,
        end=end, 
        static_lam=1,
        verbose=2,
        show_linalg_warning=False,
        fit_model_after_tuning=True
    )
    ### PREDICT WITH BEST HYPERPARAMETER(S)
    val_predictions   = cross_val_predict(best_model, X=x_train, y=y_train, cv=kfold)   
    train_predictions = best_model.predict(x_train)
    test_predictions  = best_model.predict(x_test)
    
    if anomaly:
        pass
    else:
        val_predictions   = np.maximum(val_predictions, 0)
        train_predictions = np.maximum(train_predictions, 0)
        test_predictions  = np.maximum(test_predictions, 0)
        
    print(f"""
Finish:
    F1: {f1}
    F2: {f2}
    One-hot encoding: {he}
    Anomaly: {anomaly}
    Split: {split}
    Random state: {random_state}
    N-splits: {n_splits}
    Include climate: {include_climate}
    Climate vars: {variable_groups_str}
    Final Val R2:  {r2_score(y_train, val_predictions):0.4f} 
    Final Test R2: {r2_score(y_test, test_predictions):0.4f}
    Total time: {(time.time()-tic)/60:0.2f} minutes
    """, flush=True)

    #########################################     DE-MEAN TRAIN R2    #########################################
    fold_list = []
    for i in range(n_splits):
        idx = len(list(kfold.split(y_train))[i][1])
        fold = np.repeat(i + 1, idx).tolist()
        fold_list.append(fold)
    fold_list = [item for sublist in fold_list for item in sublist]

    train_split = pd.DataFrame(
        np.repeat("train", len(x_train)), columns=["data_fold"], index=x_train.index
    )
    train_split = train_split.join(
        crop_yield.copy()[crop_yield.index.isin(x_train.index)]
    )
    train_split["oos_prediction"] = val_predictions
    train_split["val_fold"] = fold_list
    train_split = demean_by_group(train_split, predicted="oos_prediction", group=["district"])

    #########################################     DE-MEAN TEST R2    #########################################
    test_split = pd.DataFrame({"data_fold": np.repeat("test", len(x_test))}, index=x_test.index)
    test_split = test_split.join(crop_yield.copy()[crop_yield.index.isin(x_test.index)])
    test_split["oos_prediction"] = test_predictions
    test_split["val_fold"] = n_splits + 1
    test_split = demean_by_group(test_split, predicted="oos_prediction", group=["district"])

    #########################################     OUT OF SAMPLE PREDICTIONS    #########################################
    oos_preds = pd.concat([train_split, test_split])
    oos_preds[["split", "random_state"]] = split, random_state
    oos_preds["variables"] = variable_groups_str
    oos_preds["anomaly"] = anomaly
    oos_preds["hot_encode"] = he
    
    #########################################     SCORES    #########################################
    val_R2 = r2_score(y_train, val_predictions)
    val_r = pearsonr(val_predictions, y_train)[0]
    train_R2 = r2_score(y_train, train_predictions)
    train_r = pearsonr(train_predictions, y_train)[0]
    test_R2 = r2_score(y_test, test_predictions)
    test_r = pearsonr(test_predictions, y_test)[0]

    if anomaly:
        demean_cv_R2   = np.nan
        demean_cv_r    = np.nan
        demean_test_R2 = np.nan
        demean_test_r  = np.nan
    else:
        demean_cv_R2 = r2_score(train_split.demean_log_yield, train_split.demean_oos_prediction)
        demean_cv_r  = pearsonr(train_split.demean_log_yield, train_split.demean_oos_prediction)[0]
        demean_test_R2 = r2_score(test_split.demean_log_yield, test_split.demean_oos_prediction)
        demean_test_r  = pearsonr(test_split.demean_log_yield, test_split.demean_oos_prediction)[0]

    #########################################     SAVE RESULTS    #########################################
    d = {
        "split": split,
        "random_state": random_state,
        "variables": variable_groups_str,
        "anomaly": anomaly,
        "country": country_code[0],
        "year_range": yrs,
        "satellite_1": satellite1[0],
        "bands_1": bands1,
        "num_features_1": num_features1,
        "points_1": points1,
        "month_range_1": mns1,
        "limit_months_1": limit_months1,
        "crop_mask_1": crop_mask1,
        "weighted_avg_1": weighted_avg1,
        "satellite_2": satellite2[0],
        "bands_2": bands2,
        "num_features_2": num_features2,
        "points_2": points2,
        "month_range_2": mns2,
        "limit_months_2": limit_months2,
        "crop_mask_2": crop_mask2,
        "weighted_avg_2": weighted_avg2,
        "hot_encode": he,
        "total_n": len(x_all),
        "train_n": len(x_train),
        "test_n": len(x_test),
        "best_reg_param": [best_lambdas],
        "mean_of_val_R2": [best_scores],
        "val_R2": val_R2,
        "val_r": val_r,
        "val_r2": val_r ** 2,
        "train_R2": train_R2,
        "train_r": train_r,
        "train_r2": train_r ** 2,
        "test_R2": test_R2,
        "test_r": test_r,
        "test_r2": test_r ** 2,
        "demean_cv_R2": demean_cv_R2,
        "demean_cv_r": demean_cv_r,
        "demean_cv_r2": demean_cv_r ** 2,
        "demean_test_R2": demean_test_R2,
        "demean_test_r": demean_test_r,
        "demean_test_r2": demean_test_r ** 2,
    }
    if return_oos_predictions:
        return d, oos_preds
    else:
        return d
