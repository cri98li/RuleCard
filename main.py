import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from RuleCard.RuleCardGAM import RuleCardGAM

def _fit_pred_times(model, X_train, y_train, X_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    return round(end_train - start_train, 4), round(end_test - start_test, 4), y_pred

if __name__ == "__main__":
    df = pd.read_csv('datasets/CLF/diabetes.csv')
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    train_time, pred_time, y_pred = _fit_pred_times(rf, X_train, y_train, X_test)
    print('Random Forest:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')

    rcgam = RuleCardGAM()
    train_time, pred_time, y_pred = _fit_pred_times(rcgam, X_train, y_train, X_test)
    print('RuleCardGAM:', classification_report(y_test, y_pred), end='\t')
    print(f'Training time: {train_time}s\tPrediction time: {pred_time}s\n\n')
