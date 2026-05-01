# =========================================================
# 1. Library Imports
# =========================================================
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")


# =========================================================
# 2. Config
# =========================================================
RANDOM_STATE = 42
TARGET_COL = "LoanApproved"
DATA_PATH = "Loan_Approval_Final.csv"

ARTIFACT_DIR = Path("artifacts")
PLOT_DIR = ARTIFACT_DIR / "plots"

ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)
# =========================================================
# 3. Helper Functions
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered ratio features.
    Uses NaN for divide-by-zero protection.
    """
    df = df.copy()

    annual_income_safe = df["AnnualIncome"].replace(0, np.nan)
    loan_amount_safe = df["LoanAmount"].replace(0, np.nan)

    df["LoanToIncomeRatio"] = df["LoanAmount"] / annual_income_safe
    df["MonthlyDebtBurden"] = df["MonthlyDebtPayments"] / (annual_income_safe / 12)
    df["LiquidityToLoanRatio"] = (
        df["SavingsAccountBalance"] + df["CheckingAccountBalance"]
    ) / loan_amount_safe
    df["NetWorthToLoanRatio"] = df["NetWorth"] / loan_amount_safe

    return df


def drop_leakage_columns(df: pd.DataFrame):
    leakage_cols = [
        "ApplicationDate",
        "BaseInterestRate",
        "InterestRate",
        "MonthlyLoanPayment",
        "TotalDebtToIncomeRatio",
        "RiskScore",
    ]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols).copy()
    return df, drop_cols


def build_preprocessors(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    linear_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    tree_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return linear_preprocess, tree_preprocess, numeric_features, categorical_features


def build_models(linear_preprocess, tree_preprocess, y_train):
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", linear_preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", tree_preprocess),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.80,
                        min_child_weight=3,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_predictions(y_true, prob, threshold=0.50):
    pred = (prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "confusion_matrix": confusion_matrix(y_true, pred),
        "pred": pred,
        "prob": prob,
    }


def print_eval(name, results, threshold=0.50):
    print(f"\n{name}")
    print(f"Threshold : {threshold:.2f}")
    print(f"ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


def threshold_analysis(y_true, prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.30, 0.76, 0.05)

    rows = []
    for t in thresholds:
        res = evaluate_predictions(y_true, prob, threshold=t)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "accuracy": res["accuracy"],
            }
        )

    return pd.DataFrame(rows)


def choose_threshold_by_f1(threshold_df: pd.DataFrame) -> float:
    best_row = threshold_df.sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).iloc[0]
    return float(best_row["threshold"])


def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_ohe.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names


def plot_model_comparison(df_results: pd.DataFrame, save_path: str):
    plt.figure(figsize=(9, 5))
    x = np.arange(len(df_results))
    width = 0.2

    plt.bar(x - 1.5 * width, df_results["roc_auc"], width, label="ROC-AUC")
    plt.bar(x - 0.5 * width, df_results["precision"], width, label="Precision")
    plt.bar(x + 0.5 * width, df_results["recall"], width, label="Recall")
    plt.bar(x + 1.5 * width, df_results["f1"], width, label="F1")

    plt.xticks(x, df_results["model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.title("Validation Model Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(model_probs: dict, y_true, save_path: str):
    plt.figure(figsize=(7, 5))
    for model_name, prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, save_path: str, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_metrics(threshold_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    plt.title("Validation Metrics by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_boxplot(cv_scores, save_path: str):
    plt.figure(figsize=(6, 4))
    plt.boxplot(cv_scores)
    plt.title("5-Fold CV ROC-AUC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 4. Load Data
# =========================================================
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# =========================================================
# 5. Feature Engineering
# =========================================================
df = add_engineered_features(df)

# =========================================================
# 6. Leakage Prevention
# =========================================================
df, dropped_leakage_cols = drop_leakage_columns(df)

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

print("Dropped leakage columns:", dropped_leakage_cols)
print("Feature matrix shape:", X.shape)
print("Target distribution:")
print(y.value_counts())

# =========================================================
# 7. Train / Validation / Test Split
# =========================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE,
)

val_size = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=val_size,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# =========================================================
# 8. Build Preprocessors
# =========================================================
linear_preprocess, tree_preprocess, numeric_features, categorical_features = build_preprocessors(X_train)

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# =========================================================
# 9. Build Models
# =========================================================
models = build_models(linear_preprocess, tree_preprocess, y_train)

# =========================================================
# 10. Validation Model Comparison
# =========================================================
validation_rows = []
validation_probs = {}
fitted_models = {}

for model_name, pipe in models.items():
    print(f"\nTraining: {model_name}")
    pipe.fit(X_train, y_train)

    val_prob = pipe.predict_proba(X_val)[:, 1]
    val_results = evaluate_predictions(y_val, val_prob, threshold=0.50)
    print_eval(f"{model_name} (Validation)", val_results, threshold=0.50)

    validation_rows.append(
        {
            "model": model_name,
            "roc_auc": val_results["roc_auc"],
            "precision": val_results["precision"],
            "recall": val_results["recall"],
            "f1": val_results["f1"],
            "accuracy": val_results["accuracy"],
        }
    )

    validation_probs[model_name] = val_prob
    fitted_models[model_name] = pipe

validation_df = pd.DataFrame(validation_rows).sort_values(
    ["roc_auc", "f1", "recall"], ascending=False
).reset_index(drop=True)

print("\nValidation Comparison Table")
print(validation_df)

validation_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
plot_model_comparison(validation_df, PLOT_DIR / "validation_model_comparison.png")
plot_roc_curves(validation_probs, y_val, PLOT_DIR / "validation_roc_curves.png")

# =========================================================
# 11. Select Final Model
# =========================================================
best_model_name = validation_df.iloc[0]["model"]
best_model = fitted_models[best_model_name]
best_val_prob = validation_probs[best_model_name]

print("\nSelected final candidate:", best_model_name)

# =========================================================
# 12. Threshold Analysis
# =========================================================
threshold_df = threshold_analysis(y_val, best_val_prob)
best_threshold = choose_threshold_by_f1(threshold_df)

print("\nThreshold Analysis")
print(threshold_df)
print("\nBest threshold selected by validation F1:", best_threshold)

threshold_df.to_csv(ARTIFACT_DIR / "threshold_analysis.csv", index=False)
plot_threshold_metrics(threshold_df, PLOT_DIR / "threshold_analysis.png")
save_json({"threshold": best_threshold}, ARTIFACT_DIR / "threshold.json")

# =========================================================
# 13. Cross-Validation Stability Check
# =========================================================
cv_model = best_model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)

print("\n5-Fold CV ROC-AUC Scores:")
print(cv_scores)
print("Mean:", cv_scores.mean())
print("Std :", cv_scores.std())

save_json(
    {
        "cv_scores": [float(x) for x in cv_scores],
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    },
    ARTIFACT_DIR / "cv_results.json",
)

plot_cv_boxplot(cv_scores, PLOT_DIR / "cv_boxplot.png")


# =========================================================
# 14. Final Test Evaluation
# =========================================================
test_prob = best_model.predict_proba(X_test)[:, 1]
test_results = evaluate_predictions(y_test, test_prob, threshold=best_threshold)
print_eval(f"{best_model_name} (Test)", test_results, threshold=best_threshold)

print("\nClassification Report (Test)")
print(classification_report(y_test, test_results["pred"], digits=4))

plot_confusion_matrix(
    test_results["confusion_matrix"],
    PLOT_DIR / "test_confusion_matrix.png",
    title=f"Confusion Matrix ({best_model_name} Test)"
)

save_json(
    {
        "final_model_name": best_model_name,
        "test_roc_auc": float(test_results["roc_auc"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "test_accuracy": float(test_results["accuracy"]),
        "selected_threshold": float(best_threshold),
    },
    ARTIFACT_DIR / "final_test_metrics.json",
)

# =========================================================
# 15. Error Analysis
# =========================================================
test_pred = test_results["pred"]

fp_idx = X_test.index[(y_test.values == 0) & (test_pred == 1)]
fn_idx = X_test.index[(y_test.values == 1) & (test_pred == 0)]

print("\nFalse Positives:", len(fp_idx))
print("False Negatives:", len(fn_idx))

error_cols = [
    c for c in [
        "CreditScore",
        "AnnualIncome",
        "LoanAmount",
        "DebtToIncomeRatio",
        "LoanToIncomeRatio",
        "MonthlyDebtBurden",
        "LiquidityToLoanRatio",
        "NetWorthToLoanRatio",
        "EmploymentStatus",
        "EducationLevel",
        "MaritalStatus",
        "Gender",
    ]
    if c in X_test.columns
]

fp_samples = X_test.loc[fp_idx, error_cols].head(10)
fn_samples = X_test.loc[fn_idx, error_cols].head(10)

fp_samples.to_csv(ARTIFACT_DIR / "false_positive_samples.csv", index=True)
fn_samples.to_csv(ARTIFACT_DIR / "false_negative_samples.csv", index=True)

error_summary = {
    "false_positives": int(len(fp_idx)),
    "false_negatives": int(len(fn_idx)),
    "more_concerning_error_type": "False Negative",
    "reason": (
        "False negatives are more concerning when the model incorrectly rejects applicants "
        "who were actually approved in the data."
    ),
    "deployment_concern": (
        "Borderline applicants and cases with incomplete or unusual financial profiles "
        "may be more error-prone."
    )
}
save_json(error_summary, ARTIFACT_DIR / "error_analysis_summary.json")

# =========================================================
# 16. Save Final Model and Input Columns
# =========================================================
joblib.dump(best_model, ARTIFACT_DIR / "final_model.pkl")

feature_columns = X.columns.tolist()
save_json({"feature_columns": feature_columns}, ARTIFACT_DIR / "feature_columns.json")

background_sample = X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE)
background_sample.to_csv(ARTIFACT_DIR / "background_sample.csv", index=False)

# =========================================================
# 17. SHAP Explainability
# =========================================================
print("\nRunning SHAP explainability...")

final_preprocessor = best_model.named_steps["preprocess"]
final_estimator = best_model.named_steps["model"]

X_train_transformed = final_preprocessor.transform(X_train)
X_test_transformed = final_preprocessor.transform(X_test)

feature_names = get_feature_names_from_preprocessor(
    final_preprocessor,
    numeric_features,
    categorical_features,
)

if hasattr(X_train_transformed, "toarray"):
    X_train_transformed_dense = X_train_transformed.toarray()
else:
    X_train_transformed_dense = X_train_transformed

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed_dense = X_test_transformed.toarray()
else:
    X_test_transformed_dense = X_test_transformed

# SHAP for XGBoost final model
if best_model_name == "XGBoost":
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_test_transformed_dense)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_transformed_dense,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_global_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance_df.to_csv(ARTIFACT_DIR / "shap_global_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    top_shap = shap_importance_df.head(15).sort_values("mean_abs_shap")
    plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
    plt.title("Top 15 SHAP Global Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_global_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    if len(fn_idx) > 0:
        selected_index = fn_idx[0]
    else:
        selected_index = X_test.index[0]

    row_num = list(X_test.index).index(selected_index)

    local_explanation = shap.Explanation(
        values=shap_values[row_num],
        base_values=explainer.expected_value,
        data=X_test_transformed_dense[row_num],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(local_explanation, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_local_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

    local_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[row_num]
    })
    local_df["abs_shap"] = np.abs(local_df["shap_value"])
    local_df = local_df.sort_values("abs_shap", ascending=False)
    local_df.to_csv(ARTIFACT_DIR / "shap_local_case.csv", index=False)

    save_json(
        {
            "selected_case_index": int(selected_index),
            "predicted_probability": float(test_prob[row_num]),
            "predicted_class": int(test_results["pred"][row_num]),
            "actual_class": int(y_test.loc[selected_index]),
        },
        ARTIFACT_DIR / "shap_local_case_info.json",
    )
else:
    # fallback summary for non-XGBoost final model
    save_json(
        {
            "message": "Final model is not XGBoost, so Tree SHAP plots were not generated for the final model.",
            "final_model_name": best_model_name
        },
        ARTIFACT_DIR / "shap_local_case_info.json",
    )

# =========================================================
# 18. Fairness Slice Check
# =========================================================
group_list = ["Gender", "EducationLevel", "MaritalStatus", "EmploymentStatus"]
fairness_rows = []

for col in group_list:
    if col in X_test.columns:
        temp = pd.DataFrame({
            col: X_test[col],
            "Actual": y_test,
            "Pred": test_results["pred"]
        })

        temp = temp[temp[col].notna()]

        for g in temp[col].unique():
            sub = temp[temp[col] == g]

            cm = confusion_matrix(sub["Actual"], sub["Pred"], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

            fairness_rows.append({
                "group_column": col,
                "group_value": str(g),
                "count": int(len(sub)),
                "recall": float(recall) if pd.notna(recall) else None,
                "fpr": float(fpr) if pd.notna(fpr) else None,
            })

fairness_df = pd.DataFrame(fairness_rows)
fairness_df.to_csv(ARTIFACT_DIR / "fairness_summary.csv", index=False)

fairness_interpretation = {
    "checked_groups": group_list,
    "method": "Compared subgroup recall and false positive rate on the test set.",
    "important_note": (
        "Observed subgroup differences should be treated as monitoring concerns, "
        "not automatic proof of confirmed unfairness."
    ),
    "possible_causes": [
        "Different subgroup sample sizes",
        "Underlying data imbalance",
        "Feature distribution differences across subgroups"
    ],
    "deployment_guidance": (
        "Monitor subgroup performance after deployment and re-evaluate if large disparities persist."
    )
}
save_json(fairness_interpretation, ARTIFACT_DIR / "fairness_interpretation.json")

# =========================================================
# 19. Progress Summary
# =========================================================
progress_summary = {
    "version": "Post-Whiteboard Final Technical Version",
    "changes_since_whiteboard": [
        "Added engineered financial ratio features",
        "Removed leakage-related variables",
        "Expanded model comparison to include XGBoost",
        "Added threshold analysis",
        "Added SHAP global explanation",
        "Added SHAP local case explanation",
        "Added subgroup fairness check",
        "Added structured error analysis",
        "Added deployment-ready artifact saving"
    ],
    "fixed_or_removed": [
        "Removed leakage-prone columns from modeling",
        "Separated validation and final test usage",
        "Improved evaluation beyond accuracy only"
    ],
    "why_it_matters": [
        "Improves trustworthiness of evaluation",
        "Improves interpretability of final model",
        "Improves readiness for deployment and executive review"
    ]
}
save_json(progress_summary, ARTIFACT_DIR / "progress_summary.json")

# =========================================================
# 20. Model Selection Summary
# =========================================================
selection_summary = {
    "selected_model": best_model_name,
    "selection_basis": {
        "primary_metric": "Validation ROC-AUC",
        "secondary_metrics": ["Recall", "F1-score", "Precision"],
        "reason": (
            "Validation ROC-AUC was used as the main ranking metric, while recall, "
            "F1-score, and precision were used to review class-specific trade-offs."
        ),
    },
    "threshold_selection": {
        "selected_threshold": float(best_threshold),
        "basis": "Chosen using validation-set F1 with precision-recall trade-off review."
    },
    "accepted_tradeoffs": [],
    "rejected_alternatives": []
}

for _, row in validation_df.iterrows():
    if row["model"] != best_model_name:
        selection_summary["rejected_alternatives"].append({
            "model": row["model"],
            "validation_roc_auc": float(row["roc_auc"]),
            "reason_not_selected": "Lower validation performance or less favorable trade-off."
        })

if best_model_name == "Logistic Regression":
    selection_summary["accepted_tradeoffs"].append(
        "Accepted a simpler linear model because it provided strong validation performance and clearer interpretability."
    )
elif best_model_name == "XGBoost":
    selection_summary["accepted_tradeoffs"].append(
        "Accepted a more complex model because it improved predictive performance, while SHAP was used to support interpretability."
    )
elif best_model_name == "Random Forest":
    selection_summary["accepted_tradeoffs"].append(
        "Accepted a non-linear tree ensemble model, while recognizing interpretability is weaker than a linear baseline."
    )

save_json(selection_summary, ARTIFACT_DIR / "model_selection_summary.json")

# =========================================================
# 21. Deployment Readiness Summary
# =========================================================
deployment_readiness = {
    "technically_deployable_now": True,
    "recommended_positioning": "Decision-support tool, not full automation",
    "assumptions": [
        "Input features are available and correctly entered",
        "Future data follows a similar distribution to the training data",
        "The selected threshold remains appropriate for the use case"
    ],
    "monitor_after_deployment": [
        "Missing value rates",
        "Input distribution drift",
        "Approval rate drift",
        "Subgroup recall and false positive rate",
        "Probability calibration over time"
    ],
    "not_fully_production_ready_because": [
        "Real-time monitoring is not implemented in this prototype",
        "No automated retraining loop is included",
        "User-entered inputs may be incomplete or inaccurate"
    ]
}
save_json(deployment_readiness, ARTIFACT_DIR / "deployment_readiness.json")

# =========================================================
# 22. Final Freeze Summary
# =========================================================
final_freeze = {
    "finalized_components": [
        "Leakage prevention design",
        "Train/validation/test evaluation strategy",
        "Final model selection",
        "Threshold selection",
        "XAI outputs",
        "Deployment prototype structure"
    ],
    "will_not_change_before_executive_presentation": [
        "Core feature set used by the final model",
        "Final selected model",
        "Final threshold used in deployment demo"
    ],
    "executive_should_expect": [
        "A usable decision-support prototype",
        "Clear communication of model benefits, risks, and uncertainty"
    ],
    "executive_should_not_expect": [
        "A fully automated production system",
        "Perfect accuracy or zero-risk predictions"
    ]
}
save_json(final_freeze, ARTIFACT_DIR / "final_freeze_summary.json")


# =========================================================
# 23. Logistic Regression Feature Importance
# =========================================================
if best_model_name == "Logistic Regression":
    print("\nGenerating Logistic Regression feature importance...")

    final_preprocessor = best_model.named_steps["preprocess"]
    final_model = best_model.named_steps["model"]

    X_train_transformed = final_preprocessor.transform(X_train)

    feature_names = get_feature_names_from_preprocessor(
        final_preprocessor,
        numeric_features,
        categorical_features,
    )

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    coefficients = final_model.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    }).sort_values("abs_coefficient", ascending=False)

    importance_df.to_csv(ARTIFACT_DIR / "logistic_feature_importance.csv", index=False)

    top_features = importance_df.head(15).sort_values("abs_coefficient")

    plt.figure(figsize=(8,6))
    plt.barh(top_features["feature"], top_features["coefficient"])
    plt.title("Top 15 Logistic Regression Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "logistic_feature_importance.png", dpi=300)
    plt.close()

# =========================================================
# 24. Final Summary Print
# =========================================================
print("\nTraining complete.")
print("Artifacts saved to:", ARTIFACT_DIR.resolve())
print("Plots saved to:", PLOT_DIR.resolve())
print("Final selected model:", best_model_name)
print("Final threshold:", best_threshold)

