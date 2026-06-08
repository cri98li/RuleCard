import numpy as np
import math
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import Lasso, lasso_path, LassoCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

# from readers import * # Assuming this exists in your local environment

class EBMLasso(BaseEstimator, ClassifierMixin):
    def __init__(self, **ebm_params):
        self.ebm_params = ebm_params

    def fit(self, X, y):
        self.ebm_ = ExplainableBoostingClassifier(**self.ebm_params)
        self.ebm_.fit(X, y)

        contrib = self.ebm_.eval_terms(X)

        self.lasso_cv_ = LassoCV(cv=5, positive=True, random_state=42)
        self.lasso_cv_.fit(contrib, y)

        self.alpha_best = self.lasso_cv_.alpha_
        self.coef_ = self.lasso_cv_.coef_
        self.intercept_ = self.lasso_cv_.intercept_

        self._apply_pruning()
        return self

    def _apply_pruning(self):
        self.ebm_.intercept_[0] = float(self.intercept_)

        for idx, w in enumerate(self.coef_):
            self.ebm_.scale(idx, factor=w)

        self.ebm_.sweep()

    def predict(self, X):
        return self.ebm_.predict(X)

    def predict_proba(self, X):
        return self.ebm_.predict_proba(X)

    def n_trees(self):
        return int(np.sum(self.coef_ != 0))

    def n_leaves(self):
        leaves = []
        for term_idx in range(len(self.ebm_.term_names_)):
            try:
                bins = len(self.ebm_.bins_[term_idx][0])
                leaves.append(bins)
            except Exception:
                leaves.append(None)
        return leaves

    def n_leaves2(self):
        leaves = []
        for term_idx in range(len(self.ebm_.term_names_)):
            try:
                bin_structure = self.ebm_.bins_[term_idx]
                total_leaves = math.prod([len(b) for b in bin_structure])
                leaves.append(total_leaves)
            except Exception:
                leaves.append(None)
        return leaves

    def n_nodes(self):
        leaves = self.n_leaves()
        return [2 * l - 1 if l is not None else None for l in leaves]


if __name__ == '__main__':
    # _, df = read_titanic()
    # X = df.drop(columns='y').values
    # y = df.y.values

    # Synthetic fallback for independent execution
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=42)

    ebm = EBMLasso()
    ebm.fit(X_train, y_train)

    y_pred = ebm.predict(X_test).astype(float)

    print(classification_report(y_true=y_test, y_pred=y_pred))

    print(len(ebm.coef_), sum(ebm.coef_!=0))
    print('n_trees', ebm.n_trees())

    # Filter out None values before summing
    valid_leaves = [l for l in ebm.n_leaves() if l is not None]
    valid_leaves2 = [l for l in ebm.n_leaves2() if l is not None]
    valid_nodes = [n for n in ebm.n_nodes() if n is not None]

    print('n_leaves', np.sum(valid_leaves))
    print('n_leaves2', np.sum(valid_leaves2))
    print('n_nodes', np.sum(valid_nodes))