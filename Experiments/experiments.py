import copy
import hashlib
import os.path
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import dill
import h5py
import numpy as np
import pandas as pd
from RuleTree import RuleTreeRegressor, RuleTreeClassifier
from benchmark.evaluation_utils import evaluate_clf
from imodels import FIGSClassifier, TreeGAMClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from psutil import cpu_count
from pygam import LogisticGAM
from rulefit import RuleFit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from EBM_lasso import EBMLasso

#from Experiments.realkd.rules import RuleBoostingEstimator
from apriori_reg import AprioriReg
from readers import read_titanic
RuleBoostingEstimator = None

try:
    from Experiments.My_FasterRisk import MyFasterRisk
except:
    pass
from RuleCard.RuleCardGAM import RuleCardGAM
from Experiments.readers import read_bank, all_datasets
from RuleCard.feat_importance_wrapper import *

random_state = 42
n_jobs = 4

dict_hyper = {
    "log_reg": {
        "penalty": [None, 'l1', 'l2', 'elasticnet'],
        "C": [0.01, 0.1, 0.5, 1, 5, 10],
        "l1_ratio": [0.1, 0.5, 0.9],
        "dual": [False],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "max_iter": [100, 500],
        "random_state": [random_state],
        "n_jobs": [n_jobs],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1, 10, 100],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "random_state": [random_state],
    },
    "EBM": {
        "max_bins": [128, 256, 512, 1024],
        "max_interaction_bins": [16, 32, 64],
        "interactions": [0, 5, 10, "3x"],
        "validation_size": [0.15],
        "learning_rate": [0.01, 0.015, 0.05],
        "n_jobs": [n_jobs],
        "random_state": [random_state],
    },
    "FIGS": {
        "max_rules": [6, 12, 24, 48, 96],
        "max_trees": [None],
        "min_impurity_decrease": [.0],
        "random_state": [random_state],
        "max_features": [None],
        "max_depth": [None],
    },
    "LogisticGAM": {
        "terms": ['auto'],
        "max_iter": [100, 200],
        "tol": [.0001, .00001],
        "fit_intercept": [True],
    },
    "TreeGAMCl":  {
        "n_boosting_rounds": [50, 100, 200, 500],
        "max_leaf_nodes": [3],
        "reg_param": [0.0, 0.01, 0.1],
        "learning_rate": [0.005, 0.01, 0.05],
        "n_boosting_rounds_marginal": [0],
        "max_leaf_nodes_marginal": [2],
        "reg_param_marginal": [0.0],
        "fit_linear_marginal": [None],
        "boosting_strategy": ["cyclic"],
        "validation_frac": [0.15],
        "random_state": [random_state]
    },
    "RuleCard": {
        "learning_rate": [0.3, 0.5, 1.],
        "patience": [15],
        "max_depth": [1, 2, 3],
        "max_n_iter": [100, 500],
        "feature_order": ['feature_importance', 'random'], #
        "recompute_feature_order": [True, False], #
        "reuse_features": [True, False],
        "use_fast": [True, False],
        "top_k_features": [3, 5, 'sqrt', np.inf],
        "n_jobs": [n_jobs],

        "feature_importance_estimator": [
            #RuleCard_DecisionTreeRegressor(random_state=random_state),
            RuleCard_RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            #RuleCard_LGBMRegressor(random_state=random_state, n_jobs=n_jobs),
            #RuleCard_XGBoost(random_state=random_state, n_jobs=n_jobs),
            #RuleCard_CatBoostRegressor(random_state=random_state),
            #RuleCard_Variance(),
            #RuleCard_InfoGainStump(),
            #RuleCard_Chi2(),
            #RuleCard_ANOVA_regression(),
            #RuleCard_mi_regression(random_state=random_state, n_jobs=n_jobs),
        ]
    },
    'fasterrisk': {
        'sparsity': [1, 3, 5, 10, 15, 50, 100, 500, 1000]
    },
    'fasterriskBin': {
        'sparsity': [1, 3, 5, 10, 15, 50, 100, 500, 1000]
    },
    'boley': {
        'num_rules': [1, 2, 4, 10, 20, 50, 100, 200, 500]
    },
    'apriori_log': {
        'supp': [10, 20, 30, 50][::-1],
        'zmin': [10, 2],
        'quantiles': [2, 4, 8],
        'target_in_consequent': [False, True],
        'model': ['logistic'],
        'penalty': ['l2'],
        'C': [.1, .5, 1., np.inf],
        'fit_intercept': [True, False],
        'random_state': [random_state],
        'n_jobs': [n_jobs],
        'solver': ['lbfgs'],
        'confidence': [80],
        'k': [1000, 10000] # np.inf
    },
    'apriori_ridge': {
        'supp': [10, 20, 30, 50][::-1],
        'zmin': [10, 2],
        'quantiles': [2, 4, 8],
        'target_in_consequent': [False, True],
        'model': ['ridge'],
        'fit_intercept': [True, False],
        'alpha': [.0, .5, 1.0],
        'random_state': [random_state],
        'confidence': [80],
        'k': [1000, 10000] # np.inf
    },
    'rulefit': {
        'n_jobs': [n_jobs],
        'random_state': [random_state],
        'rfmode': ['classification'],
        'max_rules': [10, 50, 100, 500, 1000, 2000],
        'memory_par': [.1, .01],
        'model_type': ['r', 'l', 'rl'],
        'max_iter': [100]
    },
    'lgbm': {
        'boosting_type': ['gbdt'],
        'num_leaves': [31],
        'max_depth': [-1, 2, 3, 5, 10],
        'learning_rate': [.1],
        'n_estimators': [10, 100, 1000],
        'class_weight': [None],
        'n_jobs': [n_jobs],
        'random_state': [random_state],
    },
    'RF': {
        'max_depth': [None, 1, 2, 3, 5, 10],
        'n_estimators': [10, 100, 1000],
        'class_weight': [None],
        'n_jobs': [n_jobs],
        'random_state': [random_state],
    },
    'dt': {
        'max_depth': [None, 1, 2, 3, 5, 10],
        'random_state': [random_state],
    },
    'xgboost': {
        'n_estimators': [10, 100, 1000],
        'max_depth': [3, 5, 10],
        'learning_rate': [.1, .01, .3],
        'objective': ['binary:logistic'],
        'random_state': [random_state],
        'n_jobs': [n_jobs],
    }
}


dict_hyper['EBMLasso'] = copy.copy(dict_hyper['EBM'])
dict_hyper['EBMSweep'] = copy.copy(dict_hyper['EBM'])
#supp, zmin, quantiles, target_in_consequent=False, model='logistic'

def get_model(model_name):
    if model_name == "log_reg":
        return LogisticRegression
    elif model_name == "ridge":
        return RidgeClassifier
    elif model_name in ["EBM", "EBMSweep"]:
        return ExplainableBoostingClassifier
    elif model_name == "EBMLasso":
        return EBMLasso
    elif model_name == "FIGS":
        return FIGSClassifier
    elif model_name == "LogisticGAM":
        return LogisticGAM
    elif model_name == "TreeGAMCl":
        return TreeGAMClassifier
    elif model_name == "RuleCard":
        return lambda max_depth, **kwargs: RuleCardGAM(**kwargs, base_estimator=RuleTreeRegressor(max_depth=max_depth))
    elif model_name == 'fasterrisk':
        return MyFasterRisk
    elif model_name == 'fasterriskBin':
        return lambda sparsity: MyFasterRisk(sparsity, bin=True)
    elif model_name == 'boley':
        return RuleBoostingEstimator
    elif 'apriori_' in model_name:
        return AprioriReg
    elif model_name == 'rulefit':
        return RuleFit
    elif model_name == 'lgbm':
        return LGBMClassifier
    elif model_name == 'RF':
        return RandomForestClassifier
    elif model_name == 'dt':
        return RuleTreeClassifier
    elif model_name == 'xgboost':
        return XGBClassifier
    else:
        raise ValueError(f"Unknown model: {model_name}")

def gen_path(dataset_name, model_name, hypers):
    hyper_hash = hashlib.md5(str(hypers).encode()).hexdigest()
    return f'res/{dataset_name}/{model_name}/{hyper_hash}'

def get_rules(node, real_idx, X):
    if node.is_leaf():
        return [([], node.prediction, len(X))]

    rules = []

    feature = real_idx[node.stump.feature_original[0]]
    threshold = node.stump.threshold_original[0]
    cat = node.stump.is_categorical

    if cat:
        X_l = X[X[:, feature] == threshold]
        X_r = X[X[:, feature] != threshold]
        op_l, op_r = "==", "!="
    else:
        X_l = X[X[:, feature] <= threshold]
        X_r = X[X[:, feature] > threshold]
        op_l, op_r = "<=", ">"

    if node.node_l is not None:
        for r, pred, support in get_rules(node.node_l, real_idx, X_l):
            rules.append(([(feature, threshold, op_l)] + r, pred, support))

    if node.node_r is not None:
        for r, pred, support in get_rules(node.node_r, real_idx, X_r):
            rules.append(([(feature, threshold, op_r)] + r, pred, support))

    return rules

def get_all_rules(clf: RuleCardGAM, X):
    rules = []
    for est in clf.estimators_:
        if est[1].root.is_leaf():
            continue
        rules += get_rules(est[1].root, est[0], X)

    return rules

def run(X, y, dataset_name, model_name, hypers):
    if model_name == 'RuleCard' and hypers['use_fast'] and hypers['max_depth'] == 1:
        return 'skip'

    if 'k' in hypers and not np.isfinite(hypers['k']):
        del hypers['k']
        filename = gen_path(dataset_name, model_name, hypers)
    else:
        filename = gen_path(dataset_name, model_name, hypers)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename + '.pkl'):
        return 'skip'

    cv_folds = 5
    res = copy.copy(hypers) | {'dataset_name': dataset_name, 'model_name': model_name}
    if model_name == 'boley':
        X = pd.DataFrame(X, columns=[f'feat{i}' for i in range(X.shape[1])])
        y = pd.Series(y).astype(float).replace(0, -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state,
                                                        stratify=y)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    k_fold_metrics = {}
    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
        if model_name == 'boley':
            X_train_val, y_train_val = X_train.iloc[train_index], y_train.iloc[train_index]
            X_val, y_val = X_train.iloc[test_index], y_train.iloc[test_index]
        else:
            X_train_val, y_train_val = X_train[train_index], y_train[train_index]
            X_val, y_val = X_train[test_index], y_train[test_index]

        try:
            clf = get_model(model_name)(**hypers)

            start_time = time.time()
            clf.fit(X_train_val, y_train_val)
            stop_time = time.time()
        except ValueError as e:
            if 'l2' in str(e):
                return 'skip'
            else:
                #raise e
                return f'error: {e}'

        if 'cv_fit_time' not in res:
            res['cv_fit_time'] = .0
            res['cv_predict_time'] = .0
        res['cv_fit_time'] += stop_time - start_time

        if 'Sweep' in model_name:
            clf.sweep()

        if model_name == 'boley':
            start_time = time.time()
            y_pred_proba = clf.predict(X_val)
            stop_time = time.time()
            y_pred = [-1 if x <= 0 else 1 for x in y_pred_proba]
        else:
            start_time = time.time()
            y_pred = clf.predict(X_val)
            stop_time = time.time()
            try:
                y_pred_proba = clf.predict_proba(X_val)
            except AttributeError as e:
                y_pred_proba = None
        res['cv_predict_time'] += stop_time - start_time

        for k, v in evaluate_clf(y_val, y_pred, y_pred_proba).items():
            k = f'val_{k}'
            if k not in k_fold_metrics:
                res[k] = .0
            res[k] += v/cv_folds

    clf = get_model(model_name)(**hypers)

    start_time = time.time()
    clf.fit(X_train, y_train)
    res['fit_time'] = time.time() - start_time

    if 'Sweep' in model_name:
        clf.sweep()

    if model_name == 'boley':
        start_time = time.time()
        y_pred_proba = clf.predict(X_test)
        res['predict_time'] = time.time() - start_time
        y_pred = [-1 if x <= 0 else 1 for x in y_pred_proba]
    else:
        start_time = time.time()
        y_pred = clf.predict(X_test)
        res['predict_time'] = time.time() - start_time
        try:
            y_pred_proba = clf.predict_proba(X_test)
        except AttributeError as e:
            y_pred_proba = None
    res |= evaluate_clf(y_test, y_pred, y_pred_proba)

    try:
        pickle.dump(clf, open(filename+'.pkl', 'wb'))
    except AttributeError:
        dill.dump(clf, open(filename+'.pkl', 'wb'))

    pd.DataFrame([res]).to_csv(filename+'.csv')

    with h5py.File(filename + '.h5', 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        if model_name == 'RuleCard':
            with open(filename + '.rules', 'wb') as f:
                pickle.dump(get_all_rules(clf, X_train), f)

    return 'ok'



if __name__ == "__main__":
    n = cpu_count()//n_jobs
    processes = []

    with ProcessPoolExecutor(max_workers=n) as executor:
        for dataset in tqdm(all_datasets, position=0, leave=True, desc='Datasets'):
            dataset_name, df = dataset()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.astype(np.int64)

            if n == 1:
                for model_name in tqdm(dict_hyper.keys(), position=1, leave=False, desc=f'Fitting models for {dataset_name}'):
                    if model_name == 'boley':
                        continue # over 24h
                    if 'fasterriskB' not in model_name:
                        continue
                    for val_comb in tqdm(product(*dict_hyper[model_name].values()), position=2, leave=False):
                        diz = dict(zip(dict_hyper[model_name].keys(), val_comb))
                        res = run(X, y, dataset_name, model_name, diz)
            else:
                #processes = []
                for model_name in tqdm(dict_hyper.keys(), position=1, leave=False, desc=f'Fitting models for {dataset_name}'):
                    if model_name == 'boley':
                        continue # over 24h
                    if 'fasterrisk' not in model_name:
                        continue
                    """if model_name not in ['rulefit', 'lgbm', 'RF', 'dt', 'xgboost', 'log_reg', 'EBM', 'LogisticGAM',
                                          'ridge', 'TreeGAMCl', 'EBMLasso', 'EBMSweep'] or 'apriori' in model_name: #'apriori_ridge':
                        continue"""

                    for val_comb in tqdm(product(*dict_hyper[model_name].values()), position=2, leave=False):
                        diz = dict(zip(dict_hyper[model_name].keys(), val_comb))
                        processes.append(executor.submit(run, X, y, dataset_name, model_name, diz))

                """for p in tqdm(processes, position=1, leave=False, desc=f'Getting results for {dataset_name}'):
                    str = p.result()
                    if 'error' in str:
                        tqdm.write(str)"""

        for p in tqdm(as_completed(processes), position=1, total=len(processes), leave=True, desc=f'Getting results'):
            str = p.result()
            if 'error' in str:
                tqdm.write(str)

