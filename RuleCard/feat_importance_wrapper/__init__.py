from abc import ABC

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class RuleCard_FeatureImportanceEstimator(ABC):
    def get_feature_importance(self) -> np.ndarray:
        pass

class RuleCard_LogisticRegression(LogisticRegression, RuleCard_FeatureImportanceEstimator):
    def get_feature_importance(self):
        return self.coef_


class RuleCard_DecisionTreeRegressor(DecisionTreeRegressor, RuleCard_FeatureImportanceEstimator):
    def get_feature_importance(self):
        return self.feature_importances_


class RuleCard_RandomForestRegressor(RandomForestRegressor, RuleCard_FeatureImportanceEstimator):
    def get_feature_importance(self):
        return self.feature_importances_


class RuleCard_LGBMRegressor(LGBMRegressor, RuleCard_FeatureImportanceEstimator):
    def get_feature_importance(self):
        return self.feature_importances_


class RuleCard_XGBoost(XGBRegressor, RuleCard_FeatureImportanceEstimator):
    def get_feature_importance(self):
        return self.feature_importances_


class RuleCard_CatBoostRegressor(CatBoostRegressor, RuleCard_FeatureImportanceEstimator):
    pass


class RuleCard_Variance(RegressorMixin, RuleCard_FeatureImportanceEstimator):
    def fit(self, X, y):
        self.var = np.var(X, axis=0)
        return self

    def get_feature_importance(self):
        return self.var

class RuleCard_InfoGainStump(RegressorMixin, RuleCard_FeatureImportanceEstimator):
    def fit(self, X, y):
        gains = np.zeros(X.shape[1])

        for j in range(X.shape[1]):
            clf = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X[:, [j]], y)

            if clf.tree_.node_count > 1:
                gains[j] = clf.tree_.impurity[0] - (
                        clf.tree_.n_node_samples[1] / clf.tree_.n_node_samples[0] * clf.tree_.impurity[1] +
                        clf.tree_.n_node_samples[2] / clf.tree_.n_node_samples[0] * clf.tree_.impurity[2]
                )

        self.gains = gains
        return self

    def get_feature_importance(self):
        return self.gains

class RuleCard_Chi2(RegressorMixin, RuleCard_FeatureImportanceEstimator):
    def fit(self, X, y):
        self.chi2, self.pval = chi2(X, y)
        return self

    def get_feature_importance(self):
        return self.chi2


class RuleCard_ANOVA_regression(RegressorMixin, RuleCard_FeatureImportanceEstimator):
    def fit(self, X, y):
        self.F, self.pval = f_regression(X, y)
        return self

    def get_feature_importance(self):
        return self.F


class RuleCard_mi_regression(RegressorMixin, RuleCard_FeatureImportanceEstimator):
    def __init__(self, discrete_features='auto', n_neighbors=3, copy=True, random_state=None, n_jobs=None):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.copy = copy
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.mi = mutual_info_regression(X, y,
                                      discrete_features=self.discrete_features, n_neighbors=self.n_neighbors,
                                      copy=self.copy, random_state=self.random_state, n_jobs=self.n_jobs)
        return self

    def get_feature_importance(self):
        return self.mi

