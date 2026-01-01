import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "AUC_PR": average_precision_score(y_true, y_prob),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred)
    }


def cross_validate_model(model, X, y, k=5):

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


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        base_score=0.5,  # Explicitly set base_score
        random_state=42,
        use_label_encoder=False
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(
        xgb,
        param_grid,
        scoring="average_precision",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def save_model(model, model_name, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, path)
    return path


def run_modeling_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir="models",
    dataset_tag=None
):

    results = {}
    suffix = f"_{dataset_tag}" if dataset_tag else ""

    # Logistic Regression (baseline, no CV)
    lr_model = train_logistic_regression(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "test_metrics": evaluate_model(y_test, lr_pred, lr_prob),
        "cv_metrics": None
    }

    save_model(lr_model, f"logistic_regression{suffix}", output_dir)

    # XGBoost (main model, with CV)
    xgb_model = train_xgboost(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results["XGBoost"] = {
        "test_metrics": evaluate_model(y_test, xgb_pred, xgb_prob),
        "cv_metrics": cross_validate_model(xgb_model, X_train, y_train)
    }

    save_model(xgb_model, f"xgboost{suffix}", output_dir)

    return results


def select_best_model(results):

    rows = []

    for model_name, metrics in results.items():
        if metrics["cv_metrics"] is None:
            continue  # skip models without CV

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
        "from stratified cross-validation. "
        "Logistic Regression was evaluated only on the test set "
        "as an interpretable baseline."
    )

    return comparison_df, best_model, justification


def plot_test_metrics(results):
    rows = []

    for model_name, metrics in results.items():
        test = metrics["test_metrics"]
        rows.append({
            "Model": model_name,
            "Accuracy": test["Accuracy"],
            "Precision": test["Precision"],
            "Recall": test["Recall"],
            "F1": test["F1"],
            "AUC_PR": test["AUC_PR"]
        })

    df = pd.DataFrame(rows).set_index("Model")

    df.plot(kind="bar", figsize=(10, 6))
    plt.title("Test Set Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_xgboost_cv(results):

    if "XGBoost" not in results or results["XGBoost"]["cv_metrics"] is None:
        print("No CV results for XGBoost.")
        return

    cv = results["XGBoost"]["cv_metrics"]

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["F1_mean", "AUC_PR_mean"],
        [cv["F1_mean"], cv["AUC_PR_mean"]]
    )
    plt.ylim(0, 1)
    plt.title("XGBoost Cross-Validation Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results):

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = metrics["test_metrics"]["Confusion_Matrix"]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()
