import copy
from concurrent.futures.process import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from RuleTree import RuleTreeRegressor
from interpret.utils import measure_interactions
from joblib import Parallel, delayed
from ordered_set import OrderedSet
from scipy.special import expit
from scipy.special.cython_special import logit
from sklearn.base import ClassifierMixin, BaseEstimator
from tqdm.auto import tqdm

from RuleCard.feat_importance_wrapper import *


class RuleCardGAM(ClassifierMixin, BaseEstimator):
    def __init__(self, learning_rate=.1, patience=5, val_size=.15, base_estimator=RuleTreeRegressor(max_depth=3),
                 max_n_iter=100,
                 feature_order='feature_importance',  # feature_importance, random
                 feature_importance_estimator=RuleCard_RandomForestRegressor(max_depth=3),
                 recompute_feature_order=False,
                 reuse_features=True,  # True, False
                 top_k_features=np.inf,
                 use_fast=True,
                 n_jobs=1,
                 random_state=42, verbose=False):
        assert issubclass(feature_importance_estimator.__class__, RuleCard_FeatureImportanceEstimator)

        self.learning_rate = learning_rate
        self.patience = patience
        self.val_size = val_size
        self.base_estimator = base_estimator
        self.max_n_iter = max_n_iter
        self.feature_order = feature_order
        self.feature_importance_estimator = feature_importance_estimator
        self.reuse_features = reuse_features
        self.recompute_feature_order = recompute_feature_order
        self.top_k_features = top_k_features
        self.use_fast = use_fast
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.base_estimator.random_state = self.random_state

        self.feature_combinations_ = OrderedSet()
        self.feature_importance_ = None
        self.used_features_ = OrderedSet()

    def _orderby_feature_importance(self, X, y):
        X0 = X
        for cols in self.used_features_:
            X0[:, cols] = .0

        topk = min(X.shape[1], self.top_k_features)
        feat_importance = self.feature_importance_estimator.fit(X0, y).get_feature_importance()
        feat_importance_idx = np.argsort(feat_importance)[:topk]
        feat_importance = feat_importance[feat_importance_idx]

        return feat_importance, OrderedSet([(x, ) for x in feat_importance_idx]) | self.feature_combinations_

    def _orderby_random(self, X, y):
        topk = min(X.shape[1], self.top_k_features)
        return OrderedSet([(x, ) for x in range(X.shape[1])][:topk]) - self.used_features_

    def _fast(self, X, y):
        admissible_features = self._orderby_random(X, y)
        if len(admissible_features) < 2:
            return OrderedSet()
        interactions = measure_interactions(X, y, interactions=combinations([x[0] for x in admissible_features], 2))#[:1]

        return OrderedSet([c for c, scores in interactions])

    def _get_combinations(self, X, y):
        if not self.recompute_feature_order and self.feature_importance_ is not None:
            return self.feature_combinations_ - self.used_features_

        self.feature_importance_, self.feature_combinations_ = self._orderby_feature_importance(X, y)
        if self.use_fast:
            self.feature_combinations_ |= self._fast(X, y)

        return self.feature_combinations_ - self.used_features_

    def fit(self, X, y):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        if n_classes != 2:
            raise ValueError("RuleCardGAM only support binary classification tasks")
        y = (y == self.classes_[1]).astype(float)

        if self.top_k_features == 'sqrt':
            self.top_k_features = max(1, int(X.shape[1]**.5))

        n_val = int(self.val_size * X.shape[0])
        np.random.seed(self.random_state)
        idx = np.random.permutation(X.shape[0])
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.base_prediction_ = np.sum(y_train) / y_train.shape[0]
        self.base_log_odds = logit(self.base_prediction_)
        residuals = y_train - self.base_prediction_

        prediction = np.ones((X_train.shape[0],)) * self.base_prediction_
        prediction_val = np.ones((X_val.shape[0],)) * self.base_prediction_
        log_odds_prediction = np.ones((X_train.shape[0],)) * self.base_log_odds
        log_odds_prediction_val = np.ones((X_val.shape[0],)) * self.base_log_odds

        self.estimators_ = []
        wait = 0
        for it_idx in tqdm(range(self.max_n_iter), position=0, leave=False, disable=not self.verbose):
            best_feat_idx = (-1, )
            best_est = None
            best_gamma_map = {}
            best_res_delta, best_res_delta_val = None, None
            best_score, best_score_val = np.inf, np.inf

            if self.n_jobs == 1:
                for feat_idx in tqdm(self._get_combinations(X_train, residuals), position=1, leave=False,
                                     disable=not self.verbose):
                    score, score_val, est, feat_idx, gamma_map, res_delta, res_delta_val = _rulecardGAM_innerloop(
                        self.base_estimator, feat_idx, self.learning_rate, X_train[:, feat_idx], X_val[:, feat_idx],
                        y_train, y_val, residuals, prediction, log_odds_prediction, log_odds_prediction_val)

                    if best_score > score:
                        best_score, best_score_val = score, score_val
                        best_est = est
                        best_feat_idx = feat_idx
                        best_gamma_map = gamma_map
                        best_res_delta = res_delta
                        best_res_delta_val = res_delta_val

            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(_rulecardGAM_innerloop)(self.base_estimator, feat_idx, self.learning_rate,
                                                    X_train[:, feat_idx], X_val[:, feat_idx], y_train, y_val, residuals,
                                                    prediction, log_odds_prediction, log_odds_prediction_val)
                    for feat_idx in tqdm(self._get_combinations(X_train, residuals), position=1, leave=False, disable=not self.verbose)
                )

                for score, score_val, est, feat_idx, gamma_map, res_delta, res_delta_val in results:
                    if best_score > score:
                        best_score, best_score_val = score, score_val
                        best_est = est
                        best_feat_idx = feat_idx
                        best_gamma_map = gamma_map
                        best_res_delta = res_delta
                        best_res_delta_val = res_delta_val

            if best_feat_idx == (-1,):
                return self
            log_odds_prediction += best_res_delta
            log_odds_prediction_val += best_res_delta_val
            best_prediction = expit(log_odds_prediction)
            best_prediction_val = expit(log_odds_prediction_val)

            if best_score_val < np.mean(np.abs(y_val - prediction_val)):
                wait = 0
                prediction_val = best_prediction_val
            else:
                wait += 1
                if wait >= self.patience:
                    self.estimators_ = self.estimators_[:-wait]
                    return self
            prediction = best_prediction
            residuals = y_train - prediction

            if not self.reuse_features:
                self.used_features_.append(best_feat_idx)
            self.estimators_.append((best_feat_idx, best_est, best_gamma_map))

        return self

    def predict(self, X):
        return np.vectorize(lambda x: self.classes_[x])((self.predict_proba(X) > .5).astype(int))

    def predict_proba(self, X):
        log_odds = np.ones((X.shape[0],)) * self.base_log_odds
        for feat_idx, est, gamma_map in self.estimators_:
            leafs = est.apply(X[:, feat_idx].reshape(X.shape[0], -1))
            leafs = np.vectorize(gamma_map.get)(leafs)
            log_odds += self.learning_rate * leafs
        prediction = expit(log_odds)
        return prediction


def _rulecardGAM_innerloop(base_estimator, feat_idx, learning_rate, X_train, X_val, y_train, y_val, residuals,
                           prediction, log_odds_prediction, log_odds_prediction_val):
    est = copy.deepcopy(base_estimator)
    est.fit(X_train.reshape(X_train.shape[0], -1), residuals)

    if est.root.is_leaf():
        pass

    leafs = est.apply(X_train.reshape(X_train.shape[0], -1))
    leafs_val = est.apply(X_val.reshape(X_val.shape[0], -1))
    gamma_map = {}
    denom = np.sum(prediction * (1 - prediction))
    for leaf_id in np.unique(leafs):
        gamma_map[leaf_id] = residuals[leafs == leaf_id].sum() / denom

    gamma = np.vectorize(gamma_map.get)(leafs)
    gamma_val = np.vectorize(gamma_map.get)(leafs_val)
    res_delta = learning_rate * gamma
    res_delta_val = learning_rate * gamma_val

    new_prediction = expit(log_odds_prediction + res_delta)
    new_prediction_val = expit(log_odds_prediction_val + res_delta_val)

    score = np.mean(np.abs(y_train - new_prediction))
    if not np.isinf(score):
        pass
    score_val = np.mean(np.abs(y_val - new_prediction_val))

    return score, score_val, est, feat_idx, gamma_map, res_delta, res_delta_val