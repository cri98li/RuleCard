import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
from RuleTree import RuleTreeRegressor
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from Experiments.readers import read_telco
from RuleCard.RuleCardGAM import RuleCardGAM

def pretty_print_rules(rules):
    for rule, prediction, support in rules:
        s = ''
        for idx, thr, comp in rule:
            s += f'X[{idx}] {comp} {round(thr, 2)}\tAND\t'
        s = s[:-4] + f'=> {round(prediction, 2)} \t ({support})'
        print(s)

def get_rules(node, real_idx, X):
    if node.is_leaf():
        return [([], node.prediction, len(X))]

    rules = []

    feature = real_idx[node.stump.feature_original[0]]
    threshold = node.stump.threshold_original[0]
    cat = node.stump.is_categorical

    if cat:
        X_l = X[X[:, feature] == threshold]
        X_r = X[X[:, feature] != threshold]
        op_l, op_r = "==", "!="
    else:
        X_l = X[X[:, feature] <= threshold]
        X_r = X[X[:, feature] > threshold]
        op_l, op_r = "<=", ">"

    if node.node_l is not None:
        for r, pred, support in get_rules(node.node_l, real_idx, X_l):
            rules.append(([(feature, threshold, op_l)] + r, pred, support))

    if node.node_r is not None:
        for r, pred, support in get_rules(node.node_r, real_idx, X_r):
            rules.append(([(feature, threshold, op_r)] + r, pred, support))

    return rules

def get_all_rules(clf: RuleCardGAM, X):
    rules = []
    for est in clf.estimators_:
        if est[1].root.is_leaf():
            continue
        rules += get_rules(est[1].root, est[0], X)

    return rules


def _fit_pred_times(model, X_train, y_train, X_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    with open('test.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('test.pkl', 'rb') as f:
        model = pickle.load(f)

    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    try:
        pretty_print_rules(get_all_rules(model, X_train))
    except:
        pass

    return round(end_train - start_train, 4), round(end_test - start_test, 4), y_pred


if __name__ == "__main__":
    _, df = read_telco()
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(rf, X_train, y_train, X_test)
    print('Random Forest:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    rcgam = RuleCardGAM(n_jobs=16, use_fast=True, reuse_features=False, recompute_feature_order=True,
                        feature_order='random', max_n_iter=500, patience=15, learning_rate=0.5, top_k_features=5,
                        base_estimator=RuleTreeRegressor(max_depth=3))
    train_time, pred_time, y_pred = _fit_pred_times(rcgam, X_train, y_train, X_test)
    print('RuleCardGAM:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    RocCurveDisplay.from_predictions(y_train, rcgam.predict_proba(X_train), name='train')
    RocCurveDisplay.from_predictions(y_test, rcgam.predict_proba(X_test), name='test').plot()
    plt.show()
