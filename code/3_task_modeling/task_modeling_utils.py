import pandas as pd
import numpy as np
import warnings
import time
import pyarrow
from glum import GeneralizedLinearRegressor as glm
from scipy.linalg import LinAlgWarning
from pyhere import here

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr,  pearsonr

directory = here("data", "random_features", "summary")

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


def kfold_rr_multi_lambda_tuning(
    X, 
    y,
    grid=np.logspace(-8, 8, base = 10, num = 17), 
    n_splits=5,
    start=0, 
    end=12, 
    static_lam=1,
    verbose=True,
    show_linalg_warning=False,
    fit_model_after_tuning=True
):
    if show_linalg_warning:
        pass
    else:
        warnings.filterwarnings(action="ignore", category=LinAlgWarning, module="glum")
    
    assert (len(start) == len(end)), "Start and end indexes must have same length"
  
    kfold = KFold(n_splits=n_splits)
    alpha = {'alpha': [1]}
    
    if hasattr(start, '__iter__'):
        pass
    else:
        start = [start]; end = [end]
        
    penalties = [static_lam for j in range(X.shape[1])] 
    lambdas = []; best_scores = []
    
    for i in range(len(start)):
        
        scores = []
        
        for pen in grid:
            
            if verbose:
                print(pen, end=" ")
                
            penalties[start[i]:end[i]] = [pen for j in range(end[i]-start[i])]
            
            ridge = glm(family="normal", P2=penalties, l1_ratio=0, random_state=42)   
            search = GridSearchCV(ridge, alpha, scoring = 'r2', cv = kfold).fit(X, y)
            
            scores.append(search.best_score_)
            
        best_lambda = grid[np.argmax(scores)]
        penalties[start[i]:end[i]] = [best_lambda for j in range(end[i]-start[i])]
        
        if verbose:
            print(f'''\n\tBest \u03BB {i+1}: {best_lambda}\n\tVal R2 {i+1}: {scores[np.argmax(scores)]:0.4f}''') 
        
        lambdas.append(best_lambda)
        best_scores.append(scores[np.argmax(scores)])
        
    if fit_model_after_tuning:
        
        for k in range(len(start)):
            penalties[start[k]:end[k]] = [lambdas[k] for j in range(end[k]-start[k])]
            
        ridge = glm(family="normal", P2=penalties, l1_ratio=0, random_state=42) 
        model = GridSearchCV(ridge, alpha, scoring = 'r2', cv = kfold).fit(X, y).best_estimator_
        
    else:
        model = np.nan
        
    return(lambdas, best_scores, model)


def model_1_sensor(params, n_splits=5):
#########################################     SET PARAMS    #########################################
    file         = params[0]
    hot_encode   = params[1]
    f            = file.split(sep="_")
    satellite    = f[0]
    bands        = f[1].replace("bands-", "")
    country_code = f[2]
    points       = f[3].replace("k-points", "")
    num_features = f[4].replace("-features", "")
    yrs          = f[5].replace("yr-", "").split(sep="-")
    mns          = f[6].replace("mn-", "").split(sep="-")
    limit_months = str2bool(f[7].replace("lm-", ""))
    crop_mask    = str2bool(f[8].replace("cm-", ""))
    weighted_avg = str2bool(f[9].replace("wa-", ""))
    years        = range(int(yrs[0]), int(yrs[1])+1)
    month_range  = list(range(int(mns[0]), int(mns[1])+1))

#########################################     READ DATA    #########################################
    
    fn = f"{directory}/{file}"
    features = pd.read_feather(fn)
    features.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)

    drop_cols = ['district', 'year', 'yield_mt']

    crop_yield = features.copy().loc[:, tuple(drop_cols)]
    crop_yield["log_yield"] = np.log10(crop_yield.yield_mt.to_numpy() + 1)

########################################     HOT ENCODE    ###########################################
    if hot_encode:
        drop_cols.remove("district")
        features = pd.get_dummies(features, columns=["district"], drop_first=False)
    else:
        pass

#########################################     K-FOLD SPLIT    #########################################
    x_all = features.drop(drop_cols, axis = 1) 
    y_all = np.log10(features.yield_mt.to_numpy() + 1)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

#########################################     K-FOLD CV    ###########################################
    ### SETUP
    ridge  = Ridge()  
    kfold  = KFold(n_splits=n_splits)
    alphas = {'alpha': np.logspace(-8, 8, base = 10, num = 17)}
    tic = time.time()
    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER(S)
    if hot_encode:
        best_lambdas, best_scores, best_model = kfold_rr_multi_lambda_tuning(
            X=x_train,
            y=y_train, 
            grid=alphas.get('alpha'), 
            n_splits=n_splits,
            start=[0, x_train.shape[1]-72],
            end=[x_train.shape[1]-72, x_train.shape[1]], 
            static_lam=1,
            verbose=True,
            show_linalg_warning=False,
            fit_model_after_tuning=True
        )
    else:
        search = GridSearchCV(ridge, alphas, scoring = 'r2', cv = kfold).fit(x_train, y_train)
        best_model   = search.best_estimator_
        best_scores  = search.best_score_
        best_lambdas = best_model.alpha
    ### PREDICT WITH BEST HYPERPARAMETER(S)
    val_predictions   = cross_val_predict(best_model, X=x_train, y=y_train, cv=kfold)   
    train_predictions = best_model.predict(x_train)
    test_predictions  = best_model.predict(x_test)
    print(f"File: {file}\nOne-Hot Encoding: {True}\nTotal time: {(time.time()-tic)/60:0.2f} minutes")

#########################################     DE-MEAN R2    #########################################    
    crop_yield["prediction"] = np.maximum(best_model.predict(x_all), 0)

    train_split = pd.DataFrame(np.repeat('train', len(x_train)), columns = ['split'], index = x_train.index)
    train_split = train_split.join(crop_yield.copy()[crop_yield.index.isin(x_train.index)])
    train_split['cv_prediction'] = np.maximum(val_predictions, 0)
    train_split["demean_cv_yield"] = train_split["log_yield"]-train_split.groupby('district')['log_yield'].transform('mean')
    train_split["demean_cv_prediction"] = train_split["cv_prediction"]-train_split.groupby('district')['cv_prediction'].transform('mean')

    test_split = pd.DataFrame(np.repeat('test', len(x_test)), columns = ['split'], index = x_test.index)
    test_split = test_split.join(crop_yield.copy()[crop_yield.index.isin(x_test.index)])
    test_split['cv_prediction'] = np.repeat(np.nan, len(x_test))
    test_split["demean_cv_yield"] = np.repeat(np.nan, len(x_test))
    test_split["demean_cv_prediction"] = np.repeat(np.nan, len(x_test))

    predictions = pd.concat([train_split, test_split])

#########################################     SAVE RESULTS    #########################################
    d = {
        'country'     : country_code,
        'satellite'   : satellite,
        'bands'       : bands,
        'num_features': num_features,
        'points'      : points, 
        'month_range' : f'{min(month_range)}-{max(month_range)}',

        'limit_months': limit_months,
        'crop_mask'   : crop_mask,
        'weighted_avg': weighted_avg,
        'hot_encode': hot_encode,

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

        'demean_cv_R2': r2_score(train_split.demean_cv_yield, train_split.demean_cv_prediction),
        'demean_cv_r':  pearsonr(train_split.demean_cv_yield, train_split.demean_cv_prediction)[0],
        'demean_cv_r2': pearsonr(train_split.demean_cv_yield, train_split.demean_cv_prediction)[0] ** 2,
    }
    return pd.DataFrame(data=d, index=[0])


def model_2_sensor(params, n_splits=5):
#########################################     SET PARAMS    #########################################    
    f1         = params[0]
    f2         = params[1]
    hot_encode = params[2]

    satellite1, bands1, country_code, points1, yrs1, mns1,\
    num_features1, limit_months1, crop_mask1, weighted_avg1 = split_fn(f1)

    satellite2, bands2, country_code, points2, yrs2, mns2,\
    num_features2, limit_months2, crop_mask2, weighted_avg2 = split_fn(f2)

#########################################     READ DATA    #########################################
    features_1 = pd.read_feather(here('data', 'random_features', 'summary', f1))
    features_2 = pd.read_feather(here('data', 'random_features', 'summary', f2))
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

    n_districts = len(features.district.unique())

    crop_yield = features.copy().loc[:, tuple(drop_cols)]
    crop_yield["log_yield"] = np.log10(crop_yield.yield_mt.to_numpy() + 1)

#########################################    HOT ENCODE    ######################################### 
    if hot_encode:
        drop_cols.remove('district')
        features = pd.get_dummies(features, columns = ["district"], drop_first = False)
    else:
        pass

#########################################    STANDARDIZE FEATURES    #########################################    
    features = features.set_index(drop_cols) 
    features_scaled = StandardScaler().fit_transform(features.values)
    features = pd.DataFrame(features_scaled, index=features.index).reset_index()
    features.columns = features.columns.astype(str)

#########################################     K-FOLD SPLIT    #########################################
    x_all = features.drop(drop_cols, axis = 1) 
    y_all = np.log10(features.yield_mt.to_numpy() + 1)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

#########################################     K-FOLD CV    ###########################################
    ### SETUP
    ridge  = Ridge()  
    kfold  = KFold(n_splits=n_splits)
    alphas = {'alpha': np.logspace(-8, 8, base = 10, num = 17)}
    tic = time.time()
    ### LAMBDA INDICIES
    start = [0, features_1.shape[1]]
    end   = [features_1.shape[1], x_train.shape[1]] 
    if hot_encode:
        start.append(x_train.shape[1]-n_districts)
        end.append(x_train.shape[1]-n_districts)
        end.sort()
    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER(S)
    best_lambdas, best_scores, best_model = kfold_rr_multi_lambda_tuning(
        X=x_train,
        y=y_train, 
        grid=alphas.get('alpha'), 
        n_splits=n_splits,
        start=start,
        end=end, 
        static_lam=1,
        verbose=True,
        show_linalg_warning=False,
        fit_model_after_tuning=True
    )
    ### PREDICT WITH BEST HYPERPARAMETER(S)
    val_predictions   = cross_val_predict(best_model, X=x_train, y=y_train, cv=kfold)   
    train_predictions = best_model.predict(x_train)
    test_predictions  = best_model.predict(x_test)
    print(f"File: {file}\nOne-Hot Encoding: {True}\nTotal time: {(time.time()-tic)/60:0.2f} minutes")

#########################################     DE-MEAN R2    #########################################    
    crop_yield["prediction"] = np.maximum(best_model.predict(x_all), 0)

    train_split = pd.DataFrame(np.repeat('train', len(x_train)), columns = ['split'], index = x_train.index)
    train_split = train_split.join(crop_yield.copy()[crop_yield.index.isin(x_train.index)])
    train_split['cv_prediction'] = np.maximum(val_predictions, 0)
    train_split["demean_cv_yield"] = train_split["log_yield"]-train_split.groupby('district')['log_yield'].transform('mean')
    train_split["demean_cv_prediction"] = train_split["cv_prediction"]-train_split.groupby('district')['cv_prediction'].transform('mean')

    test_split = pd.DataFrame(np.repeat('test', len(x_test)), columns = ['split'], index = x_test.index)
    test_split = test_split.join(crop_yield.copy()[crop_yield.index.isin(x_test.index)])
    test_split['cv_prediction'] = np.repeat(np.nan, len(x_test))
    test_split["demean_cv_yield"] = np.repeat(np.nan, len(x_test))
    test_split["demean_cv_prediction"] = np.repeat(np.nan, len(x_test))

    predictions = pd.concat([train_split, test_split])

#########################################     SAVE RESULTS    #########################################
    d = {
        'country': country_code,

        'satellite_1'   : satellite1[0],
        'bands_1'       : bands1,
        'num_features_1': num_features1,
        'points_1'      : points1, 
        'month_range_1' : mns1,
        'limit_months_1': limit_months1,
        'crop_mask_1'   : crop_mask1,
        'weighted_avg_1': weighted_avg1,

        'satellite_2'   : satellite2[0],
        'bands_2'       : bands2,
        'num_features_2': num_features2,
        'points_2'      : points2, 
        'month_range_2' : mns2,
        'limit_months_2': limit_months2,
        'crop_mask_2'   : crop_mask2,
        'weighted_avg_2': weighted_avg2,

        'hot_encode': hot_encode,

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

        'demean_cv_R2': r2_score(train_split.demean_cv_yield, train_split.demean_cv_prediction),
        'demean_cv_r':  pearsonr(train_split.demean_cv_yield, train_split.demean_cv_prediction)[0],
        'demean_cv_r2': pearsonr(train_split.demean_cv_yield, train_split.demean_cv_prediction)[0] ** 2,
    }
    df = pd.DataFrame(data=d)


def model_1_sensor_anomaly(params):
#########################################     SET PARAMS    #########################################
    file         = params
    f            = file.split(sep="_")
    satellite    = f[0]
    bands        = f[1].replace("bands-", "")
    country_code = f[2]
    points       = f[3].replace("k-points", "")
    num_features = f[4].replace("-features", "")
    yrs          = f[5].replace("yr-", "").split(sep="-")
    mns          = f[6].replace("mn-", "").split(sep="-")
    limit_months = str2bool(f[7].replace("lm-", ""))
    crop_mask    = str2bool(f[8].replace("cm-", ""))
    weighted_avg = str2bool(f[9].replace("wa-", ""))
    years        = range(int(yrs[0]), int(yrs[1])+1)
    month_range  = list(range(int(mns[0]), int(mns[1])+1))

    alphas = {'alpha': np.logspace(-8, 8, base = 10, num = 17)}
    kfold  = KFold()
    logo   = LeaveOneGroupOut()
    ridge  = Ridge() 
#########################################     READ DATA    #########################################
    fn = f"{directory}/{file}"
    features = pd.read_feather(fn)

    drop_cols = ['district', 'year', 'yield_mt', 'crop_perc']

    if weighted_avg:
        drop_cols.remove("crop_perc")
    else:
        pass

#########################################     CALCULATE ANOMALY   #########################################
    features['yield_mt'] = np.log10(features['yield_mt'] + 1)
    features.set_index(['year', 'district'], inplace=True)
    var_cols = features.columns
    features = features[var_cols] - features.groupby(['district'], as_index=True)[var_cols].transform('mean')
    features.reset_index(drop=False, inplace=True)

#########################################     K-FOLD SPLIT    #########################################
    x_all = features.drop(drop_cols, axis = 1) 
    y_all = features.yield_mt
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

#########################################     K-FOLD  CV   ###########################################
    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER
    kfold_ridge_reg = GridSearchCV(ridge, alphas, scoring = 'r2', cv = kfold)
    kfold_ridge_reg.fit(x_train, y_train)
    kfold_best_model = kfold_ridge_reg.best_estimator_
    ### PREDICT - PREDICTING WITH BEST HYPERPARAMETER
    kfold_val_predictions = cross_val_predict(kfold_best_model, X = x_train, y = y_train, cv = kfold)   
    y_pred_train_k = kfold_best_model.predict(x_train)
    y_pred_test_k  = kfold_best_model.predict(x_test)

#########################################     SAVE RESULTS    #########################################
    d = {
        'country': country_code,
        'satellite': satellite,
        'bands': bands,
        'num_features': num_features,
        'points': points, 
        'month_range': f'{min(month_range)}-{max(month_range)}',
        
        'limit_months': limit_months,
        'crop_mask': crop_mask,
        'weighted_avg': weighted_avg,
        
        'kfold_total_n': len(x_all),
        'kfold_train_n': len(x_train),
        'kfold_test_n' : len(x_test),
        
        'kfold_best_reg_param': list(kfold_ridge_reg.best_params_.values())[0],
        'kfold_mean_of_val_R2s': kfold_ridge_reg.best_score_,
        'kfold_val_R2': r2_score(y_train, kfold_val_predictions),
        'kfold_val_r' : pearsonr(kfold_val_predictions, y_train)[0],
        'kfold_val_r2': pearsonr(kfold_val_predictions, y_train)[0] ** 2,
        
        'kfold_train_R2': r2_score(y_train, y_pred_train_k),
        'kfold_train_r' : pearsonr(y_pred_train_k, y_train)[0],
        'kfold_train_r2': pearsonr(y_pred_train_k, y_train)[0] ** 2,
        
        'kfold_test_R2': r2_score(y_test, y_pred_test_k),
        'kfold_test_r' : pearsonr(y_pred_test_k, y_test)[0],
        'kfold_test_r2': pearsonr(y_pred_test_k, y_test)[0] ** 2,
        
    }
    return pd.DataFrame(data=d, index=[0])


def model_2_sensor_anomaly(params):
#########################################     SET PARAMS    #########################################    
    f1         = params[0]
    f2         = params[1]

    satellite1, bands1, country_code, points1, yrs1, mns1,\
    num_features1, limit_months1, crop_mask1, weighted_avg1 = split_fn(f1)
    
    satellite2, bands2, country_code, points2, yrs2, mns2,\
    num_features2, limit_months2, crop_mask2, weighted_avg2 = split_fn(f2)

#########################################     READ DATA    #########################################
    features_1 = pd.read_feather(here('data', 'random_features', 'summary', f1))
    features_2 = pd.read_feather(here('data', 'random_features', 'summary', f2))
    
#########################################     CLEAN DATA    #########################################  
    min_year = max(min(features_1.year), min(features_2.year))
    max_year = min(max(features_1.year), max(features_2.year))
    
    features_1 = features_1[features_1.year >= min_year]
    features_2 = features_2[features_2.year >= min_year]
    
    features_1 = features_1[features_1.year <= max_year]
    features_2 = features_2[features_2.year <= max_year]
    
    features_1.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)
    features_2.drop(['crop_perc'], axis=1, errors='ignore', inplace=True)
    
#########################################     JOIN DATA    #########################################  
    drop_cols = ['district', 'year', 'yield_mt']
    
    features_1 = features_1.set_index(drop_cols).add_prefix("f1_")
    features_2 = features_2.set_index(drop_cols).add_prefix("f2_")
    
    features = features_1.join(features_2).reset_index()
    features = features[~features.isna().any(axis = 1)]

#########################################    STANDARDIZE FEATURES    #########################################    
    features = features.set_index(drop_cols) 
    features_scaled = StandardScaler().fit_transform(features.values)
    features = pd.DataFrame(features_scaled, index=features.index).reset_index()

#########################################     CALCULATE ANOMALY   #########################################  
    features['yield_mt'] = np.log10(features.yield_mt + 1)
    features.set_index(['year', 'district'], inplace=True)
    var_cols = features.columns
    features = features[var_cols] - features.groupby(['district'], as_index=True)[var_cols].transform('mean')
    features.reset_index(drop=False, inplace=True)

#########################################     K-FOLD SPLIT    #########################################    
    x_all = features.drop(drop_cols, axis=1)
    y_all = features.yield_mt
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)

#########################################     K-FOLD CV   ###########################################  
    ### SETUP
    alphas = {'alpha': np.logspace(-8, 8, base = 10, num = 17)}
    kfold  = KFold()
    ridge  = Ridge()    
    
    ### GRID SEARCH - FINDING BEST REGULARIZATION PARAMETER
    ridge_reg = GridSearchCV(ridge, alphas, scoring = 'r2', cv = kfold)
    ridge_reg.fit(x_train, y_train)
    best_model = ridge_reg.best_estimator_
    ### PREDICT - PREDICTING WITH BEST HYPERPARAMETER
    val_predictions = cross_val_predict(best_model, X = x_train, y = y_train, cv = kfold)   
    train_predictions = best_model.predict(x_train)
    test_predictions  = best_model.predict(x_test)

#########################################     SAVE RESULTS    #########################################
    d = {
        'country': country_code,
        
        'satellite_1'   : satellite1[0],
        'bands_1'       : bands1,
        'num_features_1': num_features1,
        'points_1'      : points1, 
        'month_range_1' : mns1,
        'limit_months_1': limit_months1,
        'crop_mask_1'   : crop_mask1,
        'weighted_avg_1': weighted_avg1,
        
        'satellite_2'   : satellite2[0],
        'bands_2'       : bands2,
        'num_features_2': num_features2,
        'points_2'      : points2, 
        'month_range_2' : mns2,
        'limit_months_2': limit_months2,
        'crop_mask_2'   : crop_mask2,
        'weighted_avg_2': weighted_avg2,

        'total_n': len(x_all),
        'train_n': len(x_train),
        'test_n' : len(x_test),
        
        'best_reg_param': list(ridge_reg.best_params_.values())[0],
        'mean_of_val_R2s': ridge_reg.best_score_,
        'val_R2': r2_score(y_train, val_predictions),
        'val_r' : pearsonr(val_predictions, y_train)[0],
        'val_r2': pearsonr(val_predictions, y_train)[0] ** 2,
        
        'train_R2': r2_score(y_train, train_predictions),
        'train_r' : pearsonr(train_predictions, y_train)[0],
        'train_r2': pearsonr(train_predictions, y_train)[0] ** 2,
        
        'test_R2': r2_score(y_test, test_predictions),
        'test_r' : pearsonr(test_predictions, y_test)[0],
        'test_r2': pearsonr(test_predictions, y_test)[0] ** 2,
    }
    return pd.DataFrame(data=d, index=[0])
