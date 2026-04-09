import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from Experiments.readers import *
from RuleCard.RuleCardGAM import RuleCardGAM
from RuleCard.ScoreCard import ScoreCard


def _fit_pred_times(model, X_train, y_train, X_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    return round(end_train - start_train, 4), round(end_test - start_test, 4), y_pred

def pretty_print_rules(rules):
    for rule, prediction, support, log_odds in rules:
        s = ''
        for idx, thr, comp in rule:
            s += f'X[{idx}] {comp} {round(thr, 2)}\tAND\t'
        s = s[:-4] + f'=> {round(prediction, 2)} \t ({support}) ({log_odds})'
        print(s)

if __name__ == "__main__":
    _, df = read_titanic()
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(rf, X_train, y_train, X_test)
    print('Random Forest:\n', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    rcgam = RuleCardGAM(n_jobs=16, use_fast=True, reuse_features=False, recompute_feature_order=False,
                        feature_order='random', max_n_iter=2, patience=5, learning_rate=1)
    train_time, pred_time, y_pred = _fit_pred_times(rcgam, X_train, y_train, X_test)
    print('RuleCardGAM:\n', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    scorecard = ScoreCard.from_rulecard(rcgam, X_train)
    y_pred = scorecard.predict(X_test)
    print('ScoreCard:\n', classification_report(y_test, y_pred), end='\t')
    rules = scorecard.rules
    pretty_print_rules(rules)


