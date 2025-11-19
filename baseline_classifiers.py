"""
Part 4: Baseline ML Classifiers for Real-vs-Synthetic Graph Detection.
Fast version: removes RBF SVM (too slow), adds proper scaling for linear models,
keeps tree-based models unscaled, and automatically cleans invalid values.
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    USE_XGB = False


def evaluate_model(clf, X_train, X_test, y_train, y_test, model_name=""):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds

    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.0

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion": confusion_matrix(y_test, preds).tolist()
    }


with open("data/features_combined.pkl", "rb") as f:
    X, y, meta = pickle.load(f)

X = np.asarray(X)
y = np.asarray(y)

print("Loaded features:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Cleaning NaN and inf values...")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []

# Logistic Regression (scaled)
scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train)
X_test_lr = scaler_lr.transform(X_test)

clf_lr = LogisticRegression(max_iter=5000)
results.append(evaluate_model(clf_lr, X_train_lr, X_test_lr, y_train, y_test, "Logistic Regression"))

# Linear SVM (scaled)
scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train)
X_test_svm = scaler_svm.transform(X_test)

clf_svm_lin = LinearSVC()
clf_svm_lin.fit(X_train_svm, y_train)
preds = clf_svm_lin.predict(X_test_svm)

results.append({
    "model": "Linear SVM",
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds),
    "auc": roc_auc_score(y_test, preds),
    "confusion": confusion_matrix(y_test, preds).tolist()
})

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=300, n_jobs=-1)
results.append(evaluate_model(clf_rf, X_train, X_test, y_train, y_test, "Random Forest"))

# Gradient Boosting
clf_gb = GradientBoostingClassifier()
results.append(evaluate_model(clf_gb, X_train, X_test, y_train, y_test, "Gradient Boosting"))

# XGBoost (CPU-safe)
if USE_XGB:
    clf_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",  # CPU-friendly
    )
    results.append(evaluate_model(clf_xgb, X_train, X_test, y_train, y_test, "XGBoost"))


with open("results_baselines.txt", "w") as f:
    for r in results:
        f.write(str(r) + "\n\n")

print("Baseline results saved to results_baselines.txt")
