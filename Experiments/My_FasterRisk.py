from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class MyFasterRisk(ClassifierMixin):
    def __init__(self, sparsity):
        self.sparsity = sparsity

    def fit(self, X, y, sample_weight=None):
        self.le = LabelEncoder()
        self.le.fit(y)
        rso = RiskScoreOptimizer(X=X, y=self.le.transform(y)*2-1, k=self.sparsity)
        rso.optimize()
        multiplier, intercept, coefficients = rso.get_models(model_index=0)
        self.clf = RiskScoreClassifier(multiplier=multiplier, intercept=intercept, coefficients=coefficients)

    def predict(self, X):
        return self.le.inverse_transform((self.clf.predict(X = X)+1)//2)

    def predict_proba(self, X):
        return self.clf.predict_prob(X=X)