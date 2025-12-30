import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob):
    """
    Compute evaluation metrics.
    """
    return {
        "F1": f1_score(y_true, y_pred),
        "AUC_PR": average_precision_score(y_true, y_prob),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred)
    }


def cross_validate_model(model, X, y, k=5):
    """
    Perform Stratified K-Fold cross-validation.
    """
    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=42
    )

    f1_scores = []
    auc_pr_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        f1_scores.append(f1_score(y_val, y_pred))
        auc_pr_scores.append(average_precision_score(y_val, y_prob))

    return {
        "F1_mean": np.mean(f1_scores),
        "F1_std": np.std(f1_scores),
        "AUC_PR_mean": np.mean(auc_pr_scores),
        "AUC_PR_std": np.std(auc_pr_scores),
    }


def train_logistic_regression(X_train, y_train):
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="average_precision",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def run_modeling_pipeline(X_train, X_test, y_train, y_test):
  
    results = {}

    # ---- Logistic Regression (Baseline) ----
    lr_model = train_logistic_regression(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "test_metrics": evaluate_model(y_test, lr_pred, lr_prob),
        "cv_metrics": cross_validate_model(lr_model, X_train, y_train)
    }

    # ---- Random Forest (Ensemble) ----
    rf_model = train_random_forest(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    results["Random Forest"] = {
        "test_metrics": evaluate_model(y_test, rf_pred, rf_prob),
        "cv_metrics": cross_validate_model(rf_model, X_train, y_train)
    }
     # ✅ SAVE MODEL 
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest_fraud_model.pkl")

    return results


def select_best_model(results):
   
    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "F1_mean": metrics["cv_metrics"]["F1_mean"],
            "AUC_PR_mean": metrics["cv_metrics"]["AUC_PR_mean"]
        })

    comparison_df = pd.DataFrame(rows).sort_values(
        by="AUC_PR_mean",
        ascending=False
    )

    best_model = comparison_df.iloc[0]["Model"]

    justification = (
        f"{best_model} was selected based on the highest mean AUC-PR "
        "across stratified 5-fold cross-validation. "
        "Logistic Regression is retained as a strong baseline due to "
        "its interpretability and transparency."
    )

    return comparison_df, best_model, justification
