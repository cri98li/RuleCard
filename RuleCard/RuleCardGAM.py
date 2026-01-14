import copy
from itertools import combinations

import numpy as np
from RuleTree import RuleTreeRegressor
from scipy.special import expit
from scipy.special.cython_special import logit
from sklearn.base import ClassifierMixin
from tqdm.auto import tqdm


class RuleCardGAM(ClassifierMixin):
    def __init__(self, learning_rate=.1, patience=5, val_size=.15, base_estimator=RuleTreeRegressor(max_depth=3),
                 max_n_iter=100,
                 random_state=42, verbose=False):
        self.learning_rate = learning_rate
        self.patience = patience
        self.val_size = val_size
        self.base_estimator = base_estimator
        self.max_n_iter = max_n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.base_estimator.random_state = self.random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        if n_classes != 2:
            raise ValueError("RuleCardGAM only support binary classification tasks")
        y = y == self.classes_[1]

        n_val = int(self.val_size * X.shape[0])
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

        pairs = list(combinations(range(X.shape[1]), 2))
        feature_idxs = [idx for idx in list(range(X.shape[1]))] + pairs
        self.estimators_ = []

        wait = 0
        for it_idx in tqdm(range(self.max_n_iter), position=0, leave=False, disable=not self.verbose):
            for feat_idx in tqdm(feature_idxs, position=1, leave=False, disable=not self.verbose):
                est = copy.deepcopy(self.base_estimator)
                est.fit(X_train[:, feat_idx].reshape(X_train.shape[0], -1), residuals)

                leafs = est.apply(X_train[:, feat_idx].reshape(X_train.shape[0], -1))
                leafs_val = est.apply(X_val[:, feat_idx].reshape(X_val.shape[0], -1))
                gamma_map = {}
                denom = np.sum(prediction * (1 - prediction))
                for leaf_id in np.unique(leafs):
                    gamma_map[leaf_id] = residuals[leafs == leaf_id].sum() / denom

                gamma = np.vectorize(gamma_map.get)(leafs)
                gamma_val = np.vectorize(gamma_map.get)(leafs_val)
                res_delta = self.learning_rate * gamma
                res_delta_val = self.learning_rate * gamma_val

                log_odds_prediction += res_delta
                log_odds_prediction_val += res_delta_val
                new_prediction = expit(log_odds_prediction)
                new_prediction_val = expit(log_odds_prediction_val)
                if self.verbose:
                    tqdm.write(f"""\
                    {np.mean(np.abs(y_val - new_prediction_val)), np.mean(np.abs(y_val - prediction_val))}\t
                    {np.mean(np.abs(y_val - new_prediction_val)) < np.mean(np.abs(y_val - prediction_val)),}\
                    """)

                if np.mean(np.abs(y_val - new_prediction_val)) < np.mean(np.abs(y_val - prediction_val)):
                    wait = 0
                    prediction_val = new_prediction_val
                else:
                    wait += 1
                    if wait >= self.patience*len(feature_idxs):
                        self.estimators_ = self.estimators_[:-self.patience*len(feature_idxs) + 1]
                        return self
                prediction = new_prediction
                residuals = y_train - prediction

                self.estimators_.append((feat_idx, est, gamma_map))

        return self

    def predict(self, X):
        return np.vectorize(lambda x: self.classes_[x])(self.predict_proba(X) > .5)

    def predict_proba(self, X):
        prediction = np.ones((X.shape[0],)) * self.base_prediction_
        for feat_idx, est, gamma_map in self.estimators_:
            leafs = est.apply(X[:, feat_idx].reshape(X.shape[0], -1))
            leafs = np.vectorize(gamma_map.get)(leafs)
            prediction += self.learning_rate * leafs
        return prediction
