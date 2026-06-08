import numpy as np
import pandas as pd
from fim import apriori
from sklearn.linear_model import LogisticRegression, RidgeClassifier


class AprioriReg:
    @classmethod
    def preprocess(X:np.ndarray, y:np.ndarray, q):
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])]).infer_objects()
        for col in df.columns:
            df[col] = pd.qcut(df[col], q).apply(lambda x: f'{x}|{col}')
        return df.values, y

    def _encode(self, df):
        X_cod = np.ones((max(len(df), 1), max(len(self.rules), 1)), dtype=np.bool_)

        for i, rule_list in enumerate(self.rules):
            for rule in rule_list:
                feat = rule.split('|')[-1]
                X_cod[:, i] &= (df[feat] == rule).values

        return X_cod

    def __init__(self, supp, zmin, quantiles, confidence=80, target_in_consequent=False, model='logistic', k=np.inf,
                 **kwargs): #ridge
        self.supp = supp
        self.zmin = zmin
        self.confidence = confidence
        self.quantiles = quantiles
        self.target_in_consequent = target_in_consequent
        self.model = model
        self.k = k
        self.kwargs = kwargs

    def fit(self, X, y):
        if self.target_in_consequent:
            df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=[f'feat_{i}' for i in range(X.shape[1])]+['y']).infer_objects()
        else:
            df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])]).infer_objects()

        for col in [x for x in df.columns if x != 'y']:
            df[col] = pd.qcut(df[col], min(len(df[col].unique())-1, self.quantiles), duplicates='drop')\
                .apply(lambda x: f'{x}'.replace('|','.')+f'|{col}')


        if 'y' in df.columns:
            df.y = df.y.apply(lambda x: f'{x}|y')

        self.rules = []
        rules = apriori(df.values, target="r", supp=self.supp, zmin=self.zmin, conf=self.confidence, report="aScl")
        if len(rules) != 0:
            rules_df = pd.DataFrame(rules, columns=["consequent", "antecedent", "abs_support", "%_support",
                                                    "confidence", "lift"]).sort_values(by="lift", axis=0, ascending=False)

            rules_df = rules_df[rules_df.antecedent.apply(lambda x: '|y' not in str(x))]
            if not self.target_in_consequent:
                rules_df = rules_df[rules_df.consequent.apply(lambda x: '|y' not in str(x))]

            self.rules = rules_df.antecedent.unique()
            if np.isfinite(self.k):
                self.rules = self.rules[:self.k]
        X_cod = self._encode(df)

        if self.model == 'logistic':
            self.clf = LogisticRegression(**self.kwargs)
        elif self.model == 'ridge':
            self.clf = RidgeClassifier(**self.kwargs)

        self.clf.fit(X_cod, y)
        return self

    def predict(self, X):
        X_cod = self._encode(pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])]).infer_objects())
        return self.clf.predict(X_cod)






