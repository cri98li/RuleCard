from sklearn.base import ClassifierMixin


class MyFasterRisk(ClassifierMixin):
    def __init__(self, sparsity, parent_size):
        self.sparsity = sparsity
        self.parent_size = parent_size

    def fit(self, X, y, sample_weight=None):
        rso = RiskScoreOptimizer()

    def predict(self, X):
        pass