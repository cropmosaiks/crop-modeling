import numpy as np
import warnings
import time
from sklearn.model_selection import KFold, GridSearchCV
from tqdm.contrib.itertools import product
from glum import GeneralizedLinearRegressor as glm
from scipy.linalg import LinAlgWarning

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
    static_lam=1e-16,
    verbose=True,
    show_linalg_warning=False,
    fit_model_after_tuning=True
):
    if show_linalg_warning:
        pass
    else:
        warnings.filterwarnings(action="ignore", category=LinAlgWarning, module="glum")
    
    assert (len(start) == len(end)), "Start and end indexes must have same length"
      
    tic = time.time()
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
        
    print(f"Total time: {(time.time()-tic)/60:0.2f} minutes")
    
    return(lambdas, best_scores, model)