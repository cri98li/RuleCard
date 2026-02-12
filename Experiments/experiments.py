import copy
import hashlib
import os.path
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from RuleTree import RuleTreeRegressor
from benchmark.evaluation_utils import evaluate_clf
from imodels import FIGSClassifier, TreeGAMClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm

from RuleCard.RuleCardGAM import RuleCardGAM
from Experiments.readers import read_bank, all_datasets
from RuleCard.feat_importance_wrapper import *

random_state = 42
n_jobs = 4

dict_hyper = {
    "log_reg": {
        "penalty": [None, 'l1', 'l2', 'elasticnet'],
        "C": [1.0, .5],
        "l1_ratio": [.0],
        "dual": [False],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "max_iter": [100],
        "random_state": [random_state],
        "n_jobs": [n_jobs],
    },
    "ridge": {
        "alpha": [1.],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "random_state": [random_state],
    },
    "EBM": {
        "max_bins": [1024],
        "max_interaction_bins": [64],
        "interactions": [0, "3x"],
        "validation_size": [0.15],
        "learning_rate": [0.015],
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
        "max_iter": [100],
        "tol": [.0001],
        "fit_intercept": [True],
    },
    "TreeGAMCl":  {
        "n_boosting_rounds": [100],
        "max_leaf_nodes": [3],
        "reg_param": [0.0],
        "learning_rate": [0.01],
        "n_boosting_rounds_marginal": [0],
        "max_leaf_nodes_marginal": [2],
        "reg_param_marginal": [0.0],
        "fit_linear_marginal": [None],
        "boosting_strategy": ["cyclic"],
        "validation_frac": [0.15],
        "random_state": [random_state]
    },
    "RuleCard": {
        "learning_rate": [0.1, 0.01, 0.3, 0.5, 0.8, 1.],
        "patience": [5, 15],
        "max_depth": [1, 2, 3],
        "max_n_iter": [100, 500],
        "feature_order": ['feature_importance', 'random'],
        "recompute_feature_order": [True, False],
        "reuse_features": [True, False],
        "use_fast": [True, False],
        "top_k_features": [np.inf, 5, 3, 'sqrt'],
        "n_jobs": [n_jobs],

        "feature_importance_estimator": [
            RuleCard_DecisionTreeRegressor(random_state=random_state),
            RuleCard_RandomForestRegressor(random_state=random_state),
            RuleCard_LGBMRegressor(random_state=random_state),
            RuleCard_XGBoost(random_state=random_state),
            RuleCard_CatBoostRegressor(random_state=random_state),
            RuleCard_Variance(),
            RuleCard_InfoGainStump(),
            RuleCard_Chi2(),
            RuleCard_ANOVA_regression(),
            RuleCard_mi_regression(random_state=random_state),
        ]
    }
}

def get_model(model_name):
    if model_name == "log_reg":
        return LogisticRegression
    elif model_name == "ridge":
        return RidgeClassifier
    elif model_name == "EBM":
        return ExplainableBoostingClassifier
    elif model_name == "FIGS":
        return FIGSClassifier
    elif model_name == "LogisticGAM":
        return LogisticGAM
    elif model_name == "TreeGAMCl":
        return TreeGAMClassifier
    elif model_name == "RuleCard":
        return lambda max_depth, **kwargs: RuleCardGAM(**kwargs, base_estimator=RuleTreeRegressor(max_depth=max_depth))
    else:
        raise ValueError(f"Unknown model: {model_name}")

def gen_path(dataset_name, model_name, hypers):
    hyper_hash = hashlib.md5(str(hypers).encode()).hexdigest()
    return f'res/{dataset_name}/{model_name}/{hyper_hash}'

def run(X, y, dataset_name, model_name, hypers):
    cv_folds = 5
    res = copy.copy(hypers) | {'dataset_name': dataset_name, 'model_name': model_name}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state,
                                                        stratify=y)

    filename = gen_path(dataset_name, model_name, hypers)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename+'.pkl'):
        return 'skip'

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    k_fold_metrics = {}
    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
        X_train_val, y_train_val = X_train[train_index], y_train[train_index]
        X_val, y_val = X_train[test_index], y_train[test_index]

        try:
            clf = get_model(model_name)(**hypers)

            start_time = time.time()
            clf.fit(X_train_val, y_train_val)
            stop_time = time.time()
        except Exception as e:
            if 'l2' in str(e):
                return 'skip'
            else:
                return f'error: {e}'  # raise e
        if 'cv_fit_time' not in res:
            res['cv_fit_time'] = .0
            res['cv_predict_time'] = .0
        res['cv_fit_time'] += stop_time - start_time

        start_time = time.time()
        y_pred = clf.predict(X_val)
        stop_time = time.time()
        try:
            y_pred_proba = clf.predict_proba(X_val)
        except AttributeError as e:
            y_pred_proba = None
        res['cv_predict_time'] += stop_time - start_time

        for k, v in evaluate_clf(y_val, y_pred, y_pred_proba).items():
            if k not in k_fold_metrics:
                res[k] = .0
            res[k] += v/cv_folds

    clf = get_model(model_name)(**hypers)

    start_time = time.time()
    clf.fit(X_train, y_train)
    res['fit_time'] = time.time() - start_time

    start_time = time.time()
    y_pred = clf.predict(X_test)
    res['predict_time'] = time.time() - start_time
    try:
        y_pred_proba = clf.predict_proba(X_test)
    except AttributeError as e:
        y_pred_proba = None
    res |= evaluate_clf(y_test, y_pred, y_pred_proba)

    pickle.dump(res, open(filename+'.pkl', 'wb'))
    pd.DataFrame([res]).to_csv(filename+'.csv')

    return 'ok'



if __name__ == "__main__":
    n = 2

    with ProcessPoolExecutor(max_workers=n) as executor:
        for dataset in tqdm(all_datasets, position=0, leave=False, desc='Datasets'):
            dataset_name, df = dataset()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.astype(np.int64)

            if n == 1:
                for model_name in tqdm(dict_hyper.keys(), position=1, leave=False, desc=f'Fitting models for {dataset_name}'):
                    for val_comb in tqdm(product(*dict_hyper[model_name].values()), position=2, leave=False):
                        diz = dict(zip(dict_hyper[model_name].keys(), val_comb))
                        res = run(X, y, dataset_name, model_name, diz)
            else:
                processes = []
                for model_name in tqdm(dict_hyper.keys(), position=1, leave=False, desc=f'Fitting models for {dataset_name}'):
                    for val_comb in tqdm(product(*dict_hyper[model_name].values()), position=2, leave=False):
                        diz = dict(zip(dict_hyper[model_name].keys(), val_comb))
                        processes.append(executor.submit(run, X, y, dataset_name, model_name, diz))

                for p in tqdm(processes, position=1, leave=False, desc=f'Getting results for {dataset_name}'):
                    str = p.result()
                    if 'error' in str:
                        tqdm.write(str)

