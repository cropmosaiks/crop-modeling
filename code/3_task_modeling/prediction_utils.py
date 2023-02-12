
import gc
import glob
import itertools
import os
import pickle
import shutil
import time
import warnings

import numpy as np
import pandas as pd

from scipy import linalg

# os.environ["CUDA_PATH"] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0'
# from cupy import linalg

from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from functools import reduce
import sklearn.metrics as metrics
from scipy.linalg.misc import LinAlgWarning
from sklearn.linear_model._base import _preprocess_data
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge as sklearn_ridge
from sklearn.linear_model import LinearRegression

# from torch import cuda
# GPU = cuda.is_available()

# if GPU:
#     import cupy as xp
#     from cupy import linalg

#     linalg_solve_kwargs = {}
#     asnumpy = xp.asnumpy
#     mempool = xp.get_default_memory_pool()
#     pinned_mempool = xp.get_default_pinned_memory_pool()
# else:
#     from scipy import linalg

#     linalg_solve_kwargs = {"sym_pos": True}
#     xp = np
#     asnumpy = np.asarray




def compute_metrics(true, pred, weights=None):
    """ takes in a vector of true values, a vector of predicted values. To add more metrics, 
    just add to the dictionary (possibly with a flag or when
    it is appropriate to add) """
    res = dict()

    residuals = true - pred
    res["mse"] = np.sum(residuals ** 2) / residuals.shape[0]
    res["r2_score"] = metrics.r2_score(true, pred, sample_weight=weights)

    return res



def _fill_results_arrays(
    y_train,
    y_test,
    pred_train,
    pred_test,
    model,
    intercept,
    hp_tuple,
    results_dict,
    weights_train = None,
    weights_test = None,
    hp_warning=None,
):
    """Fill a dictionary of results with the results for this particular
    set of hyperparameters.
    Args:
        y_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        pred_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        model (n_outcomes 1darray of arbitrary dtype)
        hp_tuple (tuple): tuple of hyperparameter values used in this model
        results_dict (dict): As created in solve functions, to be filled in.
    """

    n_outcomes = y_train.shape[1]
    for i in range(n_outcomes):

        # get index of arrays that we want to fill
        # first dimension is outcome, rest are hyperparams
        this_ix = (i,) + hp_tuple

        if weights_train is not None:
        # compute and save metrics
            results_dict["metrics_train"][this_ix] = compute_metrics(
                y_train[:, i], pred_train[:, i], weights = weights_train
            )
            results_dict["metrics_test"][this_ix] = compute_metrics(
                y_test[:, i], pred_test[:, i], weights = weights_test
            )
        else:
            results_dict["metrics_train"][this_ix] = compute_metrics(
                y_train[:, i], pred_train[:, i],
            )
            results_dict["metrics_test"][this_ix] = compute_metrics(
                y_test[:, i], pred_test[:, i],
            )

        # save predictions if requested
        if "y_pred_test" in results_dict.keys():
            results_dict["y_pred_train"][this_ix] = pred_train[:, i]
            results_dict["y_pred_test"][this_ix] = pred_test[:, i]

        # save model results if requested
        if "models" in results_dict.keys():
            results_dict["models"][this_ix] = model[i]
            results_dict["intercept"][this_ix] = intercept

        # save hp warnings if thats desired
        results_dict["hp_warning"][this_ix] = hp_warning

    return results_dict



def y_to_matrix(y):
    """ ensures that the y value is of non-empty dimesnion 1 """
    if (type(y) == list) or (type(y)==str) or (type(y)==float):
        y = np.array(y)
        
    input_shp = y.shape
    if len(input_shp) == 0:
        return y
    if (len(input_shp) == 1) or (input_shp[0]==1):
        y = y.reshape(-1, 1)
    return y

def get_dim_lengths(X_train, Y_train, Y_test=None):
    """ packages data dimensions into one object"""
    if Y_train.ndim == 1:
        n_outcomes = 1
    else:
        n_outcomes = Y_train.shape[1]
    n_ftrs = X_train.shape[1]
    n_obs_trn = Y_train.shape[0]

    results = [n_ftrs, n_outcomes, n_obs_trn]
    if Y_test is not None:
        results.append(Y_test.shape[0])
    return results


def _initialize_results_arrays(arr_shapes, return_preds, return_models):
    # these must be instantiated independently
    results_dict = {
        "metrics_test": np.empty(arr_shapes, dtype=dict),
        "metrics_train": np.empty(arr_shapes, dtype=dict),
    }
    if return_preds:
        results_dict["y_pred_test"] = np.empty(arr_shapes, dtype=np.ndarray)
        results_dict["y_pred_train"] = np.empty(arr_shapes, dtype=np.ndarray)
    if return_models:
        results_dict["models"] = np.empty(arr_shapes, dtype=np.ndarray)
        results_dict["intercept"] = np.empty(arr_shapes, dtype=np.ndarray)

    # for numerical precision tracking
    results_dict["hp_warning"] = np.empty(arr_shapes, dtype=object)
    results_dict["hp_warning"].fill(None)
    return results_dict

## LS: It's very weird that we sometimes use a custom ridge function and sometimes use the sklearn function.
## I assume the MOSAIKS authors have a good reason for this. If we are going to switch to all custom ridge, we
## can use this basic function:
def custom_ridge(X,y, lam, intercept=True, 
                 static_lam_val = None, 
                 static_lam_idxs=None,
                 XtX = None,
                 Xty = None,
                 X_offset= None,
                 y_offset = None,
                 ):
    """
    Fit a ridge regression model with one lambda parameter.
    
    Optionally, allow for a different hyperparameter for a subset of the X columns. 
    
    Parameters
    ----------
        X: :class:`numpy.ndarray` 
        y: :class: n x 1 `numpy.ndarray`
        intercept :bool:
            Option to include an intercept in the model
        static_lam_val :numeric: 
            Second hyperparameter option for a subset of the X columns. 
            If set to 0 then, some Xs will not be penalized
        static_lam_idxs :list-like
            Iterable set of indices that will receive the static lam val.
    """
    if X_offset is None or y_offset is None:
        X, y, X_offset, y_offset, _ = _preprocess_data(
        X, y, intercept, normalize=False)
    
    if XtX is None:
        XtX = X.T.dot(X)
    
    if Xty is None:
        Xty = X.T.dot(y)
    
    eye = lam*np.eye(XtX.shape[1], dtype=np.float64)
    
    if static_lam_val is not None:
        if static_lam_idxs is None:
            print(("Need to input an iterable set of column indexes"
                   " for the columns with a prescribed lambda"))
        for idx in static_lam_idxs:
            eye[idx,idx] = static_lam_val
            
    # model = linalg.solve(XtX + eye, Xty, sym_pos=True)
    model = linalg.solve(XtX + eye, Xty, assume_a='pos') 
    
    # model = linalg.solve(XtX + eye, Xty) 
    # model = linalg.lstsq(XtX + eye, Xty)

    intercept_term = y_offset - X_offset.dot(model)

    return model, intercept_term

def ridge_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    svd_solve=False,
    lambdas=[1e2],
    return_preds=True,
    return_model=False,
    clip_bounds=None,
    intercept=False,
    static_lam_val = None,
    static_lam_idxs = None,
    allow_linalg_warning_instances=False
):
    """Train ridge regression model for a series of regularization parameters.
    Optionally clip the predictions to bounds. Used as the default solve_function
    argument for single_solve() and kfold_solve() below.
    Parameters
    ----------
        X_{train,test} : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y_{train,test} : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        svd_solve : bool, optional
            If true, uses SVD to compute w^*, otherwise does matrix inverse for each
            lambda.
        lambdas : list of floats, optional
            Regularization values to sweep over.
        return_preds : bool, optional
            Whether to return predictions for training and test sets.
        return_model : bool, optional
            Whether to return the trained weights that define the ridge regression
            model.
        clip_bounds : array-like, optional
            If None, do not clip predictions. If not None, must be ann array of
            dimension ``n_outcomes X 2``. If any of the elements of the array are None,
            ignore that bound (e.g. if a row of the array is [None, 10], apply an upper
            bound of 10 but no lower bound).
        intercept : bool, optional
            Whether to add an unregulated intercept (or, equivalently, center the X and
            Y data).
        allow_linalg_warning_instances : bool, optional
            If False (default), track for which hyperparameters did ``scipy.linalg`` 
            raise an ill-conditioned matrix error, which could lead to poor performance.
            This is used to discard these models in a cross-validation context. If True,
            allow these models to be included in the hyperparameter grid search. Note
            that these errors will not occur when using ``cupy.linalg`` (i.e. if a GPU 
            is detected), so the default setting may give differing results across 
            platforms.
    Returns
    -------
    dict of :class:`numpy.ndarray`
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance 
                metrics for each lambda
        If ``return_preds``, the following arrays will be appended in order:
            ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is itself a 1darray of {Out-of,In}-sample predictions for 
                each lambda. Each 1darray contains n_obs_{test,train} values
        if return_model, the following array will be appended:
            ``models`` : array of dimension n_outcomes X n_lambdas:
                Each element is itself a 1darray of model weights for each lambda. Each 
                1darray contains n_ftrs values
    """
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_lambdas = len(lambdas)

    # center data if needed
    X_train, y_train, X_offset, y_offset, _ = _preprocess_data(
        X_train, y_train, intercept, normalize=False
    )

    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_lambdas), return_preds, return_model
    )

    t1 = time.time()

    # send to GPU if available
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # precomputing large matrices to avoid redundant computation
    if svd_solve:
        # precompute the SVD
        U, s, Vh = linalg.svd(X_train, full_matrices=False)
        V = Vh.T
        UT_dot_y_train = U.T.dot(y_train)
    else:
        XtX = X_train.T.dot(X_train)
        Xty = X_train.T.dot(y_train)

    # iterate over the lambda regularization values
    for lx, lambdan in enumerate(lambdas):
        
        # train model
        if svd_solve:
            s_lambda = s / (s ** 2 + lambdan * np.ones_like(s))
            model = (V * s_lambda).dot(UT_dot_y_train)
            lambda_warning = None
            intercept_term = y_offset - X_offset.dot(model)
            
            if static_lam_val is not None:
                raise NotImplementedError
        else:
            with warnings.catch_warnings(record=True) as w:
                # bind warnings to the value of w
                warnings.simplefilter("always")
                lambda_warning = False
                model, intercept_term = custom_ridge(
                    None,None,lam=lambdan,intercept=intercept, # No need to input X and Y
                    XtX = XtX, Xty= Xty, #Directly feed XtX and Xty to avoid repitition
                    X_offset = X_offset, y_offset=y_offset,
                    static_lam_val = static_lam_val,
                    static_lam_idxs = static_lam_idxs)

                # if there is a warning
                if len(w) > 1:
                    for this_w in w:
                        print(this_w.message)
                    # more than one warning is bad
                    raise Exception("warning/exception other than LinAlgWarning")
                if len(w) > 0:
                    # if it is a linalg warning
                    if w[0].category == LinAlgWarning:
                        print("linalg warning on lambda={0}: ".format(lambdan))
                        # linalg warning
                        if not allow_linalg_warning_instances:
                            print("we will discard this model upon model selection")
                            lambda_warning = True
                        else:
                            lambda_warning = None
                            print("we will allow this model upon model selection")
                    else:
                        raise Exception("warning/exception other than LinAlgWarning")
        

        #####################
        # compute predictions
        #####################

        
        # send to gpu if available
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        y_offset = np.asarray(y_offset)
        X_offset = np.asarray(X_offset)

        
        # train
        pred_train = X_train.dot(model) + y_offset
        pred_train = y_to_matrix(pred_train)

        # test
        pred_test = X_test.dot(model) + intercept_term
        pred_test = y_to_matrix(pred_test)
        
        # clip if needed
        if clip_bounds is not None:
            for ix, i in enumerate(clip_bounds):
                # only apply if both bounds aren't None for this outcome
                if not (i == None).all():
                    pred_train[:, ix] = np.clip(pred_train[:, ix], *i)
                    pred_test[:, ix] = np.clip(pred_test[:, ix], *i)
                    

        # bring back to cpu if needed
        pred_train, pred_test = np.asarray(pred_train), np.asarray(pred_test)
        y_train, y_test, model, intercept_term = (
            y_to_matrix(np.asarray(y_train)),
            y_to_matrix(np.asarray(y_test)),
            y_to_matrix(np.asarray(model)),
            y_to_matrix(np.asarray(intercept_term))
        )

        # create tuple of lambda index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (lx,)

        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T

        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            intercept_term,
            hp_tuple,
            results_dict,
            hp_warning=lambda_warning,
        )

    return results_dict



def kfold_solve_custom_split_col(
    X,
    y,
    locations,
    split_col,
    sample_weights=None,
    solve_function=ridge_regression,
    num_folds=5,
    return_preds=True,
    return_model=False,
    fit_model_after_tuning=True,
    random_state=0,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.
    Args:
        X (n_obs X n_ftrs 2darray): Feature matrix
        y (n_obs X n_outcomes 2darray): Attribute matrix
        locations (n_obs): location identifiers
        split_col (n_obs X grouping 2darray): Grouping column on which to split X
        sample_weights (n_obs X grouping 2darray): Weights to be used in the regression. 
            Only implemented for the sklearn ridge solve (`solve = sklearn_ridge_regression`).
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression
            model?
        fit_model_after_tuning (bool): Fit a final model using all folds of the data
        random_state (int): Set the random state (seed) prior to splitting data
        kwargs_solve (dict): Parameters to pass to the solve func
    Returns:
        Dict of ndarrays.
            The dict will always start with the following 4 key:value pairs. "..."
                refers to a number of dimensions equivalent to the number of
                hyperparameters, where each dimension has a length equal to the number
                of values being tested for that hyperparameter. The number of
                hyperparameters and order returned is defined in the definition of the
                particular solve function we have passed as the solve_function argument:
                    metrics_test: n_folds X n_outcomes X ... ndarray of dict:
                        Out-of-sample model performance metrics for each fold, for each
                        outcome, for each hyperparameter value
                    metrics_train: n_folds X n_outcomes X ... ndarray of dict: In-sample
                        model performance metrics
                    obs_test: n_folds X  n_outcomes  X ... array of ndarray of float64:
                        Out-of-sample observed values for each fold
                    obs_train: n_folds X  n_outcomes X ... array of ndarray of float64:
                        In-sample observed values
                    cv: :py:class:`sklearn.model_selection.KFold` : kfold
                        cross-validation splitting object used
            If return_preds, the following arrays will included:
                preds_test: n_folds X  n_outcomes X ... ndarray of ndarray of float64:
                    Out-of-sample predictions or each fold, for each outcome, for each
                    hyperparameter value
                preds_train: n_folds X n_outcomes X ... ndarray of ndarray of float64:
                    In-sample predictions
            if return_model, the following array will be included:
                models: n_folds X n_outcomes X ... ndarray of same type as model: Model
                    weights/parameters. xxx here is of arbitrary dimension specific to
                    solve_function
    """
    assert num_folds > 1
    if sample_weights is not None and solve_function is not sklearn_ridge_regression:
        raise NotImplementedError("Sample weights are only implemented with the sklearn solve function")
    
    # If pandas inputs, convert to array
    X = np.array(X)
    y = np.array(y)
    locations = np.array(locations)
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    # keep track of all runs over several iterations
    kfold_metrics_test = []
    kfold_metrics_train = []
    kfold_preds_test = []
    kfold_preds_train = []
    kfold_y_train = []
    kfold_y_test = []
    kfold_models = []
    kfold_intercepts = []
    hp_warnings = []
    
    test_numeric_idxs = []
    locations_test = []
    
    print("on fold (of {0}): ".format(num_folds), end="")
    i = 0
    split_col_unique = np.unique(split_col)
    for train_grp_num, val_grp_num in kf.split(split_col_unique):
        print(i+1, end="\n")
        i += 1
        
        
        train_grp = split_col_unique[train_grp_num]
        val_grp = split_col_unique[val_grp_num]
        
        train_idxs = np.in1d(split_col, train_grp)
        val_idxs = np.in1d(split_col, val_grp)
        
        num_indxs = np.where(val_idxs)[0]
        test_numeric_idxs.append(num_indxs)
        locations_test.append(locations[num_indxs])
        
        

        X_train, X_val = X[train_idxs], X[val_idxs]
        y_train, y_val = y[train_idxs], y[val_idxs]

        if sample_weights is not None:
            weight_train, weight_val = sample_weights[train_idxs], sample_weights[val_idxs]
            kwargs_solve["weights_train"] = weight_train
            kwargs_solve["weights_val"] = weight_val

        # record train/test obs for this split
        kfold_y_train.append(y_train)
        kfold_y_test.append(y_val)

        # call solve func
        solve_results = solve_function(
            X_train,
            X_val,
            y_train,
            y_val,
            return_preds=return_preds,
            return_model=return_model,
            **kwargs_solve,
        )

        # record performance metrics
        kfold_metrics_test.append(solve_results["metrics_test"])
        kfold_metrics_train.append(solve_results["metrics_train"])

        # record optional preds and model parameters
        if return_preds:
            kfold_preds_test.append(solve_results["y_pred_test"])
            kfold_preds_train.append(solve_results["y_pred_train"])
        if return_model:
            kfold_models.append(solve_results["models"])
            kfold_intercepts.append(solve_results["intercept"])

        # recpord np warnings
        hp_warnings.append(solve_results["hp_warning"])

    # Return results
    print("\n")
    rets = {
        "metrics_test": np.array(kfold_metrics_test, dtype=object),
        "metrics_train": np.array(kfold_metrics_train,dtype=object),
        "y_true_test": np.array(kfold_y_test,dtype=object),
        "y_true_train": np.array(kfold_y_train,dtype=object),
        "hp_warning": np.array(hp_warnings,dtype=object),
        "cv": kf,
    }
    
    rets["shuffeled_num_test_indxs"] = np.array(test_numeric_idxs,dtype=object)
    rets["locations_test"] = np.array(locations_test,dtype=object)
    
    rets["solver_kwargs"] = kwargs_solve

    if return_preds:
        rets["y_pred_test"] = np.array(kfold_preds_test,dtype=object)
        rets["y_pred_train"] = np.array(kfold_preds_train,dtype=object)

    if return_model:
        rets["models"] = np.array(kfold_models,dtype=object)
        rets["intercepts"] = np.array(kfold_intercepts,dtype=object)

    if fit_model_after_tuning:
        if solve_function is not ridge_regression:
            print( ("Fit model after tuning has only been "
             "implemented with the ridge solve function"
             " no final model will be fit"))
        else:
            best_lambda_idx = interpret_kfold_results(rets, "r2_score")[0][0][0]
            best_lambda = kwargs_solve["lambdas"][best_lambda_idx]
            intercept = kwargs_solve.get("intercept", False)

            rets["prediction_model_weights"], rets["prediction_model_intercept"] = custom_ridge(
                X,y, lam=best_lambda, 
                intercept=intercept,
                static_lam_val = kwargs_solve.get("static_lam_val"),
                static_lam_idxs = kwargs_solve.get("static_lam_idxs"))

    return rets



solver_kwargs = {
# set of possible hyperparameters to search over in cross-validation
"lambdas": [1e-3,1e-2, 1e-1, 1e0,1e1,1e2,1e3,1e4,1e5,1e6],
# do you want to return the predictions from the model?
"return_preds": True,
# input the bounds used to clip predictions
"return_model": True,

# do you want to use an SVD solve or standard linear regression? (NB: SVD is much slower)
"svd_solve": False,
# do you want to allow hyperparameters to be chosen even if they lead to warnings about matrix invertibility?
"allow_linalg_warning_instances": True,

"intercept": True}




def _get_best_hps(hps, best_idxs):
    best_hps = []
    hp_names = [h[0] for h in hps]
    n_outcomes = 1
    for ox in range(n_outcomes):
        this_best_hps = []
        for hpx, hp in enumerate(best_idxs[ox]):
            this_best_hps.append(hps[hpx][1][hp])
        best_hps.append(this_best_hps)
    hp_names = np.array(hp_names)
    best_hps = np.array(best_hps)
    return best_hps, hp_names
    

def find_best_hp_idx(
    kfold_metrics_outcome, hp_warnings_outcome, crit, minimize=False, val=None, suppress_warnings= False,
):
    """Find the indices of the best hyperparameter combination,
    as scored by 'crit'.
    
    Args:
        kfold_metrics_outcome (n_folds X n_outcomes x ... ndarray of dict (hps are last)): 
            Model performance array produced by kfold_solve for a
            single outcome -- so n_outcomes must be 1
        hp_warnings_outcome (nfolds x n_hps) bool array of whether an hp warning occured
        crit (str): Key of the dicts in kfold_metrics that you want
            to use to score the model performance
        minimize (bool): If true, find minimum of crit. If false (default)
            find maximum
        val (str or int) : If not None, will tell you what outcome is being evaluated
            if a warning is raised (e.g. hyperparameters hitting search grid bounds)
        suppress_warnings (bool): If True, warning that hyperparam is at edge of range will be suppressed
            
    Returns:
        tuple of int: Indices of optimal hyperparameters. 
            Length of tuple will be equal to number of
            hyperparameters, i.e. len(kfold_metrics_test.shape) - 2
    """
    if suppress_warnings:
        sup = False
    else:
        sup = "warn"

    if minimize:
        finder = np.min
    else:
        finder = np.max

    # allowable idxs is wherever a None or a False appears in hp_warnings_outcome
    no_hp_warnings = (hp_warnings_outcome != True).all(axis=0)
    didnt_record_hp_warnings = (hp_warnings_outcome == None).any(axis=0)
    allowable_hp_idxs = np.where(no_hp_warnings == True)[0]

    assert (
        len(allowable_hp_idxs) > 0
    ), "all of your hp indices resulted in warnings so none is optimal"

    # only work with the ones that you can actually use
    kfold_metrics_outcome = kfold_metrics_outcome[:, allowable_hp_idxs]

    # get extract value for selected criteria
    # from kfold_metrics array for this particular outcome
    def extract_crit(x):
        return x[crit]

    f = np.vectorize(extract_crit)
    vals = f(kfold_metrics_outcome)

    # average across folds
    means = vals.mean(axis=0)

    # get indices of optimal hyperparam(s) - shape: num_hps x num_optimal_hp_settings
    idx_extreme = np.array(np.where(means == finder(means)))

    # warn if hp hit the bounds of your grid search for any hp (there may be >1 hps)
    for ix, this_hp in enumerate(idx_extreme):
        n_hp_vals = means.shape[ix]
        # if there's just one hp parameter, just throw one warning.
        if n_hp_vals == 1:
            print(
                "Only one value for hyperparameter number {0} supplied.".format(ix),
                  #we dont want to suppress this warning
            )
        else:
            # otherwise check if the optimal hp value is on the boundary of the search
            if 0 in this_hp:
                if didnt_record_hp_warnings[allowable_hp_idxs[ix]]:
                    print(
                        "The optimal hyperparameter is the lowest of those supplied "
                        + "(it was not checked for precision warnings). "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        ),
                    )

                else:
                    print(
                        "The optimal hyperparameter is the lowest of the acceptable (i.e. no precision warnings) "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        )
                        + "For reference, {0} of {1} ".format(
                            len(allowable_hp_idxs), len(no_hp_warnings)
                        )
                        + "hyperparamters are considered acceptable; "
                        + "their indices  are {0}.".format(allowable_hp_idxs),
                       
                    )

            if (n_hp_vals - 1) in this_hp:
                if didnt_record_hp_warnings[allowable_hp_idxs[ix]]:
                    print(
                        "The optimal hyperparameter is the highest of those supplied "
                        + "(it was not checked for precision warnings). "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        ), 
                    )
                else:
                    print(
                        "The optimal hyperparameter is the highest of the acceptable (i.e. no precision warnings) "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        )
                        + "For reference, {0} of {1} ".format(
                            len(allowable_hp_idxs), len(no_hp_warnings)
                        )
                        + "hyperparameters are considered acceptable; "
                        + "their indices  are {0}.".format(allowable_hp_idxs),
                       
                    )

    # warn if multiple optimal sets of hp
    if idx_extreme.shape[1] > 1:
        print( #We don't want to suppress this warning
            "Multiple optimal hyperparameters found for outcome {0}. Indices: {1}".format(
                val, idx_extreme
            ), 
        )

    # select first optimal set
    return tuple(allowable_hp_idxs[idx_extreme[:, 0]])


def get_fold_results_by_hp_idx(kfold_metrics, idxs):
    """Slice model performance metrics array by 
    hyperparameter indices.
    
    Args:
        kfold_metrics (n_folds X n_outcomes X ... ndarray of dict): 
            Model performance array produced by kfold_solve
        idxs (list of tuple): The indices of the hyperparameter values
            swept over in cross-validation. The dimension of the list
            indexes n_outcomes and the dimension of the tuples index ...
            
    Returns:
        n_folds X n_outcomes: Model performance for each fold using the
            set of hyperparameters defined in idxs
    """

    # initialize array of size n_folds X n_outcomes
    res = np.empty(kfold_metrics.shape[:2], dtype=kfold_metrics.dtype)

    for outcome_ix, i in enumerate(idxs):
        # slice this outcome plus the optimal hp's for this outcome
        # (first column is across folds, don't slice)
        slicer = [slice(None), outcome_ix] + list(i)
        res[:, outcome_ix] = kfold_metrics[tuple(slicer)]
    return res


def interpret_kfold_results(
    kfold_results, crits, minimize=False, save_weight_path=None, 
    save_weight_fname=None, hps=None
):
    """Return the parsed results of the best performing model from 
    kfold_solve.
    
    Args: 
        kfold_results (dict): As returned by kfold_solve()
        crits (str or list of str): Metric criteria to base contractions
            off of for each dimension. Must be str or list of length n_outcomes
        minimize (bool or list of bool) : Whether to find minimal (True) or maximal
            (False) value of each crit. (e.g. should be False for r2 and True for MSE)
        save_weight_path (optional, str): Path where weights of model should be saved 
            (if not None). Should end in '.npz'. This file will have 3 arrays. 'weights'
            will be n_folds X n_outcomes X n_features. 'hps' will be n_outcomes X n_hyperparams.
            'hp_names' will be n_hyperparams and will have the hyperparemeter names in the same
            order as the values appearing in 'hps'.
        hps (list of 2-tuples): List of hyperparameters tested. Order of the tuples is
            the same as the order they appear in kfold_results. e.g. [('lambda',[0,1,10])].
            Required if save_weight_path is not None so that the optimal HP can be saved with
            the weights.
    Returns: 
        list of tuples: The indices of the best hyperparameters for each outcome. The dimension of
            the list indexes outcomes, the dimension of the tuple indexes hyperparameters.
            If more than one hyperparameter was swept over, this inner dimension will be >1.
            In that case, the order is the same order that was used in the dimensions of 
            kfold_metrics arrays output by the solve function used to generate these results.
        n_folds X n_outcomes 2darray of dict: Model performance array for optimal set of 
            hyperparameters for each outcome across folds
        n_folds X n_outcomes 2darray of 1darray: Model predictions array for optimal set of 
            hyperparameters for each outcome across folds
    """
    kfold_metrics = kfold_results["metrics_test"]
    kfold_preds = kfold_results["y_pred_test"]
    kfold_hp_warnings = kfold_results["hp_warning"]

    if save_weight_path is not None:
        kfold_models = kfold_results["models"]

    kfold_shp = kfold_metrics.shape
    num_folds = kfold_shp[0]
    num_outputs = kfold_shp[1]

    if isinstance(minimize, bool):
        minimize = [minimize for i in range(num_outputs)]
    if isinstance(crits, str):
        crits = [crits for i in range(num_outputs)]

    best_idxs = []
    for j in range(num_outputs):
        this_output_results_by_fold = kfold_metrics.take(indices=j, axis=1)
        this_hp_warnings_by_fold = kfold_hp_warnings.take(indices=j, axis=1)
        best_idxs_for_this_output = find_best_hp_idx(
            this_output_results_by_fold,
            this_hp_warnings_by_fold,
            crits[j],
            minimize=minimize[j],
            val=j,
        )
        best_idxs.append(best_idxs_for_this_output)

    # using the indices of the best hp values, return the model performance across all folds
    # using those hp values
    metrics_best_idx = get_fold_results_by_hp_idx(kfold_metrics, best_idxs)

    # using the indices of the best hp values, return the model predictions across all folds
    # using those hp values
    y_pred_best_idx = get_fold_results_by_hp_idx(kfold_preds, best_idxs)

    # using the indices of the best hp values, return the model weights across all folds
    # using those hp values
    if all([save_weight_path, save_weight_fname]):
        os.makedirs(save_weight_path, exist_ok=True)
        best_hps, hp_names = _get_best_hps(hps, best_idxs)
        models_best_idx = get_fold_results_by_hp_idx(kfold_models, best_idxs)
        np.savez(
            os.path.join(save_weight_path, save_weight_fname),
            weights=models_best_idx, hps=best_hps, hp_names=hp_names
        )

    # return the best idx along with the metrics and preds for all the folds corresponding to that index.
    return best_idxs, metrics_best_idx, y_pred_best_idx


def get_pred_truth_locs(kfold_results):

    if type(kfold_results) is dict:
        kfold_results = [kfold_results]

    best_preds = np.vstack(
    [interpret_kfold_results(r, crits="r2_score")[2] for r in kfold_results]
        )

# flatten over fold predictions
    preds = np.vstack([y_to_matrix(i) for i in best_preds.squeeze()])

    truth = np.vstack(
    np.hstack([[y_to_matrix(i) 
                for i in r["y_true_test"].flatten().squeeze()]
        for r in kfold_results])
    )

    locations = np.hstack(kfold_results[0]["locations_test"])

    return preds, truth, locations

####### plotting

def make_train_pred_scatterplot(task, y_test, preds_test, x_label = "RCF preds", additional_title_text = "", verbose=True, alpha =.2):
    """
    Create standard scatterplot figure comparing test predictions to truths.
    
    Parameters
    ----------
    task : str
        String to be appended to the scatter title
    y_test : array or list (1 dimension)
        Truth values
    preds_test : array or list (1 dimension)
        Predicted values
    x_label : str
        String such that the prediction axis can have a new label
    additional_title_text : str
        Text that can optionally be appended to the title of the scatterplot 
    verbose: bool
        If True R2 and Pearson R are printed after being calculated.
        
    Returns
    -------
    None
    
    """
    
    test_r2 = metrics.r2_score(y_test,preds_test)
    test_pearson_r = np.corrcoef(y_test, preds_test)[0,1]
    test_spearman_r = spearmanr(y_test, preds_test).correlation
    
    if verbose:
        print('holdout r2: {0:.3f}'.format(test_r2))
        print('holdout Pearson R: {0:.3f}'.format(test_pearson_r))
        print("\n")

    fig, ax = plt.subplots( figsize=(10,10))
    ax.scatter(preds_test, y_test, alpha=alpha)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel(x_label)
    ax.set_ylabel('labels (truth)')
    title_str = task + "\n" + "$R^2$ = {0:.3f} \n Pearson R = {1:.3f}".format(test_r2,test_pearson_r)
    title_str = title_str + "\n" + "Spearman R = {0:.3f}".format(test_spearman_r)
    
    ax.set_title(additional_title_text+title_str)
    

    ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color='black')  
    
    
    
    
    
def weighted_groupby(df, groupby_col_name, weights_col_name, weight_sum_colname = None, cols_to_agg="ALL COLUMNS"):
    """
    Applies a weighted groupby to a dataframe where the weights are given in a dataframe column.
    
    Currently, this is used to area or population weight a large dataframe of features.
    
    weight_sum_colname is an optional parameter. When included, there is an additional column in the output that gives the sum of the weights for that polygon or groupby item. This allows for chunking and future re-weighting.
    
     Parameters
    ----------
    df : pd.DataFrame
        dataframe with that will receive a weighted grouping
    groupby_col_name : str or list of strings
        Will be passed as the first arg to pd.DataFrame.groupby function
    weights_col_name: str
        The column name that contains the weights for aggregating the dataframe
    weight_sum_colname : (optional) str
        If included, the weights will be saved in an output column. This string is the column name.
    cols_to_agg : (optional) list of column names
        List of columns that will receive the weighting and will be output. Default is all columns other than the weighting column and the groupby column name.
        
    Returns
    -------
    out : pd.DataFrame
    """
    df_cols = list(df.columns)
    
    assert groupby_col_name in df_cols
    assert weights_col_name in df_cols
    
    if cols_to_agg == "ALL COLUMNS":
        cols_to_agg = df_cols
        cols_to_agg.remove(groupby_col_name)
        cols_to_agg.remove(weights_col_name)
    
    if len(df) < 1: #if df is blank, return blank frame with expected colnames
        print("df < 1 ... returning blank dataframe")
        if weight_sum_colname:
            cols_to_agg += [weight_sum_colname]
            df[weight_sum_colname] = []

        return df[cols_to_agg]
        
    else:
        for col in cols_to_agg:
            assert col in df_cols
            
    def weighted(x, cols, weights=weights_col_name):
        return pd.Series(np.average(x[cols], weights=x[weights], axis=0), cols) 
    g = df.groupby(groupby_col_name)
    out = g.apply(weighted, cols_to_agg)
    
#     print("\n")
#     print("First stage weight results")
#     print(out.head())
    if weight_sum_colname:
        
        sums = g[weights_col_name].sum().rename( weight_sum_colname)
#         print("count col group results")
#         print(sums.head)
        out = pd.concat([sums,out], axis=1)
    
    return out



def sklearn_ridge_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    weights_train = None,
    weights_val = None,
    svd_solve=False,
    lambdas=[1e2],
    return_preds=True,
    return_model=False,
    clip_bounds=None,
    intercept=False,
    allow_linalg_warning_instances=False
):
    """Train ridge regression model for a series of regularization parameters.
    This uses the sklearn implementation of the ridge model, but should behave identically
    to the `ridge_regression()` function. The primary advantage of this implementation is
    that we can use the sample weight argument of the sklearn function, effectively 
    allowing for easy use of GLS in our existing CV pipeline. 

    Optionally clip the predictions to bounds. Used as the default solve_function
    argument for single_solve() and kfold_solve() below.
    Parameters
    ----------
        X_{train,test} : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y_{train,test} : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        weights_{train,val} : :class:`numpy.ndarray`
            Weights for training/test data (n_obs_{train,test} X n_ftrs 2darray). 
            Models will be weighted and R2 performance will also be weighted.
        svd_solve : bool, optional
            If true, uses SVD to compute w^*, otherwise does matrix inverse for each
            lambda.
        lambdas : list of floats, optional
            Regularization values to sweep over.
        return_preds : bool, optional
            Whether to return predictions for training and test sets.
        return_model : bool, optional
            Whether to return the trained weights that define the ridge regression
            model.
        clip_bounds : array-like, optional
            If None, do not clip predictions. If not None, must be ann array of
            dimension ``n_outcomes X 2``. If any of the elements of the array are None,
            ignore that bound (e.g. if a row of the array is [None, 10], apply an upper
            bound of 10 but no lower bound).
        intercept : bool, optional
            Whether to add an unregulated intercept (or, equivalently, center the X and
            Y data).
        allow_linalg_warning_instances : bool, optional
            If False (default), track for which hyperparameters did ``scipy.linalg`` 
            raise an ill-conditioned matrix error, which could lead to poor performance.
            This is used to discard these models in a cross-validation context. If True,
            allow these models to be included in the hyperparameter grid search. Note
            that these errors will not occur when using ``cupy.linalg`` (i.e. if a GPU 
            is detected), so the default setting may give differing results across 
            platforms.
    Returns
    -------
    dict of :class:`numpy.ndarray`
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance 
                metrics for each lambda
        If ``return_preds``, the following arrays will be appended in order:
            ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is itself a 1darray of {Out-of,In}-sample predictions for 
                each lambda. Each 1darray contains n_obs_{test,train} values
        if return_model, the following array will be appended:
            ``models`` : array of dimension n_outcomes X n_lambdas:
                Each element is itself a 1darray of model weights for each lambda. Each 
                1darray contains n_ftrs values
    """
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_lambdas = len(lambdas)

    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_lambdas), return_preds, return_model
    )

    t1 = time.time()

    # send to GPU if available
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
        
    # iterate over the lambda regularization values
    for lx, lambdan in enumerate(lambdas):
        with warnings.catch_warnings(record=True) as w:
            # bind warnings to the value of w
            warnings.simplefilter("always")
            lambda_warning = False
            m = sklearn_ridge(fit_intercept=intercept,alpha=lambdan).fit(
                X_train, y_train, sample_weight=weights_train)

            model = m.coef_

            # if there is a warning
            if len(w) > 1:
                for this_w in w:
                    print(this_w.message)
                # more than one warning is bad
                raise Exception("warning/exception other than LinAlgWarning")
            if len(w) > 0:
                # if it is a linalg warning
                if w[0].category == LinAlgWarning:
                    print("linalg warning on lambda={0}: ".format(lambdan))
                    # linalg warning
                    if not allow_linalg_warning_instances:
                        print("we will discard this model upon model selection")
                        lambda_warning = True
                    else:
                        lambda_warning = None
                        print("we will allow this model upon model selection")
                else:
                    print(w[0].message)
                    #raise Exception("warning/exception other than LinAlgWarning")
    

        #####################
        # compute predictions
        #####################

        
        # send to gpu if available
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        
        # train
        pred_train = m.predict(X_train)
        pred_train = y_to_matrix(pred_train)

        # test
        intercept_term = m.intercept_
        pred_test = m.predict(X_test)
        pred_test = y_to_matrix(pred_test)
        
        # clip if needed
        if clip_bounds is not None:
            for ix, i in enumerate(clip_bounds):
                # only apply if both bounds aren't None for this outcome
                if not (i == None).all():
                    pred_train[:, ix] = np.clip(pred_train[:, ix], *i)
                    pred_test[:, ix] = np.clip(pred_test[:, ix], *i)
                    

        # bring back to cpu if needed
        pred_train, pred_test = np.asarray(pred_train), np.asarray(pred_test)
        y_train, y_test, model, intercept_term = (
            y_to_matrix(np.asarray(y_train)),
            y_to_matrix(np.asarray(y_test)),
            y_to_matrix(np.asarray(model)),
            y_to_matrix(np.asarray(intercept_term))
        )

        # create tuple of lambda index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (lx,)

        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T

        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            intercept_term,
            hp_tuple,
            results_dict,
            weights_train = weights_train,
            weights_test = weights_val,
            hp_warning=lambda_warning,
        )

    return results_dict


############# GENERAL USE FUNCTIONS ############# 

def fix_geoid(x, str_length=5):
    assert type(x) is str
    while len(x)<str_length:
        x = "0"+x
    return x

############# FIXED EFFECTS FUNCTIONS #################
def demean(df, var_cols, group_col="GEOID", weights_col = None, return_long_grouped=False):
    
    df = df.copy()
    if weights_col is None:
        grouped = df.groupby(group_col)[var_cols].mean()
    else:
        grouped = weighted_groupby(df, group_col, weights_col, cols_to_agg=var_cols)
    
    long_grouped = df[[group_col]].merge(grouped, "left", left_on=group_col, right_index=True)
    
    if return_long_grouped:
        return long_grouped
    
    df[var_cols] = df[var_cols] - long_grouped[var_cols]
    return df


def time_group_trend_demean(df, var_cols, group_col="State", time_col = "Year", polynomial_order=1, verbose=False):
    df = df.reset_index()
    
    time_dfs = []
    
    df[time_col] = pd.to_numeric(df[time_col])
    
    for group in df[group_col].unique():
        subset = df[df[group_col]==group].copy()
        
        if type(var_cols) is str:
            var_cols = [var_cols]
        
        if len(subset) < 2:
            warnings.warn("group in time trend has fewer than two obs. Continuing without time trend demean for this group...")
            continue
            
            
        for i in range(polynomial_order):
            p = i+1
            if p == 1:
                time_Xs = subset[[time_col]]
            else:
                polynomial_col = (subset[time_col] ** p).rename(time_col+str(p))
                time_Xs = pd.concat([time_Xs, polynomial_col],axis=1)
                
            
        lm = LinearRegression(fit_intercept=True).fit(time_Xs,subset[var_cols])
        projected_time_trend = lm.predict(time_Xs)
        
        if verbose:
            print(group)
            print("time intercept:")
            print(lm.intercept_)
            print("time coefs:")
            print(lm.coef_)
            print("\n")
            
            # Remove the time trend, but do not remove the mean of the group (so add back)
        subset[var_cols] = subset[var_cols]-projected_time_trend + subset[var_cols].mean()
        time_dfs.append(subset)
    
    return pd.concat(time_dfs).set_index("index").sort_index()


def mfe(df, var_cols, group_cols, time_group=None, weights_col=None, n_iter=4, verbose=False, time_trend_polynomial=1):
    """
    MFE = multiple fixed effects. This function is designed to remove many fixed effects from a dataset through iteration.
    
    It is loosely based off the stata regxfe package that is described in:
    https://journals.sagepub.com/doi/epdf/10.1177/1536867X1501500318
    
     Parameters
    ----------
        df : pandas.DataFrame object - 
            The same input dataframe that will be returned with centered var_cols columns
    var_cols  : list
        A list of column names that will be centered via iterative demeaning. Typically the y and X vars used in the model.
    group_cols  : list
        A list of column names that correspond to each group/entity fixed effect. A list of length two will mean we demean by 
        two way FE.
    time_group  : list
        A list of tuples used for time:group fixed effects. Each tuple should be of length two. 
        The first item in the tuple is the continous time variable. The second variable is the categorical variable.
    n_iter  : int 
        The number of iterations in which the demeaning is done. 
        Initial experiments show that 3-6 iterations is typically sufficient. 
    Verbose  : bool
        Print statements throughout the iterative demeaning procedure
    weights_col : Column in df to be used for a weighted demeaning. This functionality is only implemented 
        for group/entity effects. Time:group functionality with weights is not yet implemented.
    time_trend_polynomial  :  list of int > 0 
        Use this agument to demean with polynomial time trends. 1=linear time trends, 2=quadratic, etc.
        Each list item corresponds to the time_group input order.
        WARNING - polynomial time trends should typically be normalized. This function, as currently implemented,
        does not normalize the input time variable.
    
    """
    df = df.copy()
    
    tg_error_message =  ("time group should be a list of tuples with the continuous variable"
                         "first and the time trend second")
    if isinstance(group_cols, str):
        group_cols = [group_cols]
        
    if isinstance(time_trend_polynomial, int):
        time_trend_polynomial = [time_trend_polynomial]
        
    if isinstance(time_trend_polynomial, list):
        assert len(time_trend_polynomial) == len(time_group), "ensure len(time_group) == len(time_trend_polynomial)"
        for item in time_trend_polynomial:
            assert (item > 0), "polynomial order should be greater than 0"
            assert isinstance(item, int), "polynomial order should be an integer"
    else:
        raise Exception("time_trend_polynomial should be a list of integers or a single integer")
        
    if group_cols == None or group_cols==False: ## Make the entity FE an empty list if no group cols are provided
        group_cols = []
    # time - group syntax checks
    if time_group is not None:
        assert type(time_group) is list, tg_error_message
        assert all([len(item) == 2 for item in time_group]), tg_error_message
        
        if weights_col is not None:
            raise NotImplementedError
    
    if isinstance(weights_col, list):
        assert len(weights_col)==1
        weights_col = weights_col[0]
        
    for j in range(n_iter):
        if verbose:
            
            print("n_iter=", j+1)
        for i, group_col in enumerate(group_cols):
            if verbose:
                print(f"group{i}=", group_col)
            df = demean(df,var_cols, group_col=group_col, weights_col=weights_col)
            gc.collect()
            
        if time_group is not None:
            for k, item in enumerate(time_group):
                cont, cat = item
                if verbose:
                    print(f"time_group{k}=", cont + ":"+ cat)
                df = time_group_trend_demean(df,var_cols, group_col=cat,time_col=cont, 
                                             polynomial_order=time_trend_polynomial[k], verbose=verbose)
                
                gc.collect()
    
    return df