"""
train_model.py
--------------
Trains multiple regression models on the King County house price dataset,
compares them, selects the best, and saves the full sklearn Pipeline + metadata
to models/model.pkl.

Run:
    python train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---- Paths -------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "modified_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Load data ---------------------------------------------------------------
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ---- Feature engineering -----------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["sale_year"]  = df["date"].dt.year
df["sale_month"] = df["date"].dt.month
df.drop(columns=["date"], inplace=True)

# Drop leakage / identifier columns
df.drop(columns=["price_per_sqft", "street"], inplace=True)

# ---- Target & features -------------------------------------------------------
TARGET = "price"
df = df[df[TARGET] > 0].copy()

y = df[TARGET]
X = df.drop(columns=[TARGET])

# Log-transform target: reduces the effect of extreme outliers for tree models
y_log = np.log1p(y)

# ---- Column types ------------------------------------------------------------
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "sale_year", "sale_month",
]
CATEGORICAL_FEATURES = ["city", "statezip"]

print(f"  Numeric features    : {NUMERIC_FEATURES}")
print(f"  Categorical features: {CATEGORICAL_FEATURES}")
print(f"  Target              : {TARGET}  (regression)")

# ---- Preprocessor ------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
])

# ---- Train / test split -------------------------------------------------------
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
_, _, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---- Candidate models ---------------------------------------------------------
# All models are trained on log(price) then predictions are expm1'd back
CANDIDATES = {
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    ),
}

# ---- Evaluate candidates -----------------------------------------------------
results = {}
print("\n-- Model Comparison (trained on log-price, evaluated on original price) --")
for name, estimator in CANDIDATES.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
    # 5-fold CV on log-price
    cv_scores = cross_val_score(pipe, X_train, y_train_log, cv=5, scoring="r2", n_jobs=-1)
    pipe.fit(X_train, y_train_log)

    y_pred_log = pipe.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_pred     = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "pipe":       pipe,
        "cv_r2_mean": cv_scores.mean(),
        "test_r2":    r2,
        "test_mae":   mae,
        "test_rmse":  rmse,
    }
    print(
        f"  {name:25s}  CV-R2={cv_scores.mean():.4f}  "
        f"Test-R2={r2:.4f}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}"
    )

# ---- Select best model --------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["test_r2"])
best_info = results[best_name]
print(f"\n[BEST] Best model: {best_name}  (Test R2 = {best_info['test_r2']:.4f})")

# ---- Save pipeline + log-transform flag ---------------------------------------
# We wrap the pipeline with the log/expm1 bookkeeping info in metadata
joblib.dump(best_info["pipe"], MODEL_PATH)
print(f"   Pipeline saved to {MODEL_PATH}")

# ---- Save metadata (read by app.py) ------------------------------------------
metadata = {
    "model_name":           best_name,
    "target":               TARGET,
    "problem_type":         "regression",
    "log_transform_target": True,
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features":         NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "test_r2":   round(best_info["test_r2"],  4),
    "test_mae":  round(best_info["test_mae"],  2),
    "test_rmse": round(best_info["test_rmse"], 2),
    "cities":    sorted(df["city"].dropna().unique().tolist()),
    "statezips": sorted(df["statezip"].dropna().unique().tolist()),
    "feature_ranges": {
        col: {
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in NUMERIC_FEATURES
    },
    "all_model_results": {
        name: {
            "cv_r2_mean": round(info["cv_r2_mean"], 4),
            "test_r2":    round(info["test_r2"],    4),
            "test_mae":   round(info["test_mae"],    2),
            "test_rmse":  round(info["test_rmse"],   2),
        }
        for name, info in results.items()
    },
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata saved to  {META_PATH}")
print("\nTraining complete.")
