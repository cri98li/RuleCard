import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class AprioriClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, min_support=0.1, min_confidence=0.7, bins=3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.bins = bins
        self.rules = None
        self.items = None
        self.model = LogisticRegression()
        self.bin_edges_ = {}
        self.cat_values_ = {}

    def _binarize_tabular(self, X: pd.DataFrame, fit=True):
        transactions = []
        for _, row in X.iterrows():
            items = []
            for col in X.columns:
                val = row[col]
                if np.issubdtype(type(val), np.number):
                    if fit:
                        _, edges = pd.cut(X[col], bins=self.bins, retbins=True, labels=False, duplicates='drop')
                        self.bin_edges_[col] = edges
                    else:
                        edges = self.bin_edges_[col]
                    bin_idx = np.digitize([val], edges, right=False)[0] - 1
                    bin_idx = max(0, min(bin_idx, len(edges)-2))
                    items.append(f"{col}={bin_idx}")
                else:
                    if fit:
                        self.cat_values_[col] = sorted(X[col].unique())
                    if val in self.cat_values_[col]:
                        items.append(f"{col}={val}")
            transactions.append(items)
        if fit:
            self.items = sorted(set(item for trans in transactions for item in trans))
        df_bin = pd.DataFrame([{item: (item in trans) for item in self.items} for trans in transactions])
        return df_bin

    def _rules_matrix(self, df_bin):
        matrix = np.zeros((len(df_bin), len(self.rules)), dtype=int)
        for i, (_, rule) in enumerate(self.rules.iterrows()):
            antecedents = list(rule['antecedents'])
            matrix[:, i] = df_bin[antecedents].all(axis=1).astype(int)
        return matrix

    def fit(self, X: pd.DataFrame, y):
        df_bin = self._binarize_tabular(X, fit=True)
        frequent_itemsets = apriori(df_bin, min_support=self.min_support, use_colnames=True)
        self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        X_rules = self._rules_matrix(df_bin)
        self.model.fit(X_rules, y)
        return self

    def predict(self, X: pd.DataFrame):
        df_bin = self._binarize_tabular(X, fit=False)
        X_rules = self._rules_matrix(df_bin)
        return self.model.predict(X_rules)

    def predict_proba(self, X: pd.DataFrame):
        df_bin = self._binarize_tabular(X, fit=False)
        X_rules = self._rules_matrix(df_bin)
        return self.model.predict_proba(X_rules)

    