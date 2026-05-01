# =========================================================
# 1. Library Imports
# =========================================================
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =========================================================
# 2. Config
# =========================================================
RANDOM_STATE = 42
TARGET_COL = "LoanApproved"
DATA_PATH = "Loan_Approval_Final.csv"

ARTIFACT_DIR = Path("artifacts_demo")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

SELECTED_FEATURES = [
    "CreditScore",
    "AnnualIncome",
    "LoanAmount",
    "LoanDuration",
    "MonthlyDebtPayments",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "EmploymentStatus",
    "PreviousLoanDefaults",
    "BankruptcyHistory",
]


# =========================================================
# 3. Helper Functions
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def build_preprocessor(X_train: pd.DataFrame):
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


def build_models(linear_preprocess, tree_preprocess):
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
                        n_estimators=300,
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
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


# =========================================================
# 4. Load Data
# =========================================================
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# =========================================================
# 5. Leakage Prevention
# =========================================================
df, dropped_leakage_cols = drop_leakage_columns(df)

required_cols = SELECTED_FEATURES + [TARGET_COL]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df[required_cols].copy()

y = df[TARGET_COL].astype(int)
X = df[SELECTED_FEATURES].copy()

print("Dropped leakage columns:", dropped_leakage_cols)
print("Using selected features:", SELECTED_FEATURES)
print("Feature matrix shape:", X.shape)
print("Target distribution:")
print(y.value_counts())

# =========================================================
# 6. Train / Validation / Test Split
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
# 7. Build Preprocessors
# =========================================================
linear_preprocess, tree_preprocess, numeric_features, categorical_features = build_preprocessor(X_train)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# =========================================================
# 8. Build Models
# =========================================================
models = build_models(linear_preprocess, tree_preprocess)

# =========================================================
# 9. Validation Model Comparison
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

validation_df.to_csv(ARTIFACT_DIR / "demo_model_comparison.csv", index=False)

# =========================================================
# 10. Select Final Model
# =========================================================
best_model_name = validation_df.iloc[0]["model"]
best_model = fitted_models[best_model_name]
best_val_prob = validation_probs[best_model_name]

print("\nSelected final demo model:", best_model_name)

# =========================================================
# 11. Threshold Analysis
# =========================================================
threshold_df = threshold_analysis(y_val, best_val_prob)
best_threshold = choose_threshold_by_f1(threshold_df)

print("\nThreshold Analysis")
print(threshold_df)
print("\nBest threshold selected by validation F1:", best_threshold)

threshold_df.to_csv(ARTIFACT_DIR / "demo_threshold_analysis.csv", index=False)
save_json({"threshold": best_threshold}, ARTIFACT_DIR / "demo_threshold.json")

# =========================================================
# 12. Final Test Evaluation
# =========================================================
test_prob = best_model.predict_proba(X_test)[:, 1]
test_results = evaluate_predictions(y_test, test_prob, threshold=best_threshold)
print_eval(f"{best_model_name} (Test)", test_results, threshold=best_threshold)

print("\nClassification Report (Test)")
print(classification_report(y_test, test_results["pred"], digits=4))

save_json(
    {
        "final_model_name": best_model_name,
        "test_roc_auc": float(test_results["roc_auc"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "test_accuracy": float(test_results["accuracy"]),
        "selected_threshold": float(best_threshold),
        "confusion_matrix": test_results["confusion_matrix"].tolist(),
    },
    ARTIFACT_DIR / "demo_test_metrics.json",
)

# =========================================================
# 13. Save Final Artifacts
# =========================================================
joblib.dump(best_model, ARTIFACT_DIR / "demo_model.pkl")

save_json(
    {"feature_columns": SELECTED_FEATURES},
    ARTIFACT_DIR / "demo_feature_columns.json"
)

save_json(
    {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    },
    ARTIFACT_DIR / "demo_feature_types.json"
)

print("\nDemo training complete.")
print("Artifacts saved to:", ARTIFACT_DIR.resolve())
print("Final selected demo model:", best_model_name)
print("Final demo threshold:", best_threshold)