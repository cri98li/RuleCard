from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


from typing import Optional, Tuple
import numpy as np
import pandas as pd


def nan_onehot_single_column(column: pd.Series) -> np.ndarray:
    onehot = np.zeros(len(column), dtype=int)
    onehot[column.isnull()] = 1
    return onehot


class ContinuousBinarizer:
    def __init__(
            self,
            max_num_thresholds_per_feature: int = 100,
            sampling_weights: str = "uniform",
            sampling_seed: int = 0,
    ):
        self.max_num_thresholds_per_feature = max_num_thresholds_per_feature
        self.sampling_weights = sampling_weights
        self.rng = np.random.default_rng(sampling_seed)

    def fit(self, df: pd.DataFrame) -> "ContinuousBinarizer":
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.columns_ = list(df.columns)
        self.thresholds_ = {}  # col -> list thresholds
        self.has_nan_ = {}     # col -> bool

        for col in self.columns_:
            s = df[col]

            has_nan = s.isnull().any()
            self.has_nan_[col] = has_nan

            if has_nan:
                s = s.dropna()

            if len(s.unique()) <= 1:
                self.thresholds_[col] = []
                continue

            uniq = s.unique()
            counts = s.value_counts()

            k = self.max_num_thresholds_per_feature
            if has_nan:
                k -= 1

            k = min(k, len(uniq))

            p = np.ones(len(uniq)) / len(uniq)
            if self.sampling_weights == "weighted":
                p = counts / counts.sum()
            elif self.sampling_weights != "uniform":
                raise ValueError("sampling_weights must be 'uniform' or 'weighted'")

            idx = self.rng.choice(len(uniq), k, replace=False, p=p)
            thresholds = np.sort(uniq[idx])

            # last one is not used for <= features (same as original logic)
            self.thresholds_[col] = thresholds[:-1]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("Call fit first")

        if list(df.columns) != self.columns_:
            raise ValueError("Columns mismatch between fit and transform")

        n = len(df)
        out = {}

        for col in self.columns_:
            s = df[col]

            # NaN feature
            if self.has_nan_.get(col, False):
                out[f"{col}_isNaN"] = nan_onehot_single_column(s)
                s_filled = s.fillna(0)
            else:
                s_filled = s

            thresholds = self.thresholds_.get(col, [])

            for t in thresholds:
                name = f"{col}<={t}"
                out[name] = (s_filled <= t).astype(int).values

        return pd.DataFrame(out)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

class MyFasterRisk(ClassifierMixin):
    def __init__(self, sparsity, bin=False):
        self.sparsity = sparsity
        self.bin = bin

    def fit(self, X, y, sample_weight=None):
        if self.bin:
            self.cb = ContinuousBinarizer()
            X = self.cb.fit_transform(pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])).values
        self.le = LabelEncoder()
        self.le.fit(y)
        rso = RiskScoreOptimizer(X=X, y=self.le.transform(y)*2-1, k=self.sparsity)
        rso.optimize()
        multiplier, intercept, coefficients = rso.get_models(model_index=0)
        self.clf = RiskScoreClassifier(multiplier=multiplier, intercept=intercept, coefficients=coefficients)

    def predict(self, X):
        if self.bin:
            X = self.cb.transform(pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])).values
        return self.le.inverse_transform((self.clf.predict(X = X)+1)//2)

    def predict_proba(self, X):
        if hasattr(self, 'bin') and self.bin:
            X = self.cb.transform(pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])).values
        return self.clf.predict_prob(X=X)