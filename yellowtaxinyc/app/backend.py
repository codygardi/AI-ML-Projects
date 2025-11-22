# backend.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

# --------------------------------------------------------------------
# Paths and artifact loading
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT_DIR / "deployment"


def load_artifacts():
    """
    Load scaler + sklearn models + feature column order from deployment/.
    """
    scaler_path = DEPLOY_DIR / "scaler.pkl"
    fare_model_path = DEPLOY_DIR / "fare_model_sklearn.pkl"
    tip_model_path = DEPLOY_DIR / "tip_model_sklearn.pkl"
    feature_cols_path = DEPLOY_DIR / "feature_columns.pkl"

    scaler = joblib.load(scaler_path)
    fare_model = joblib.load(fare_model_path)
    tip_model = joblib.load(tip_model_path)

    if feature_cols_path.exists():
        feature_columns = joblib.load(feature_cols_path)
    else:
        feature_columns = None

    return scaler, fare_model, tip_model, feature_columns


# --------------------------------------------------------------------
# Data prep utilities
# --------------------------------------------------------------------
def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for NYC Yellow Taxi TLC data.
    Keeps only columns needed for prediction.
    """
    df = df.copy()

    needed_cols = [
        "tpep_pickup_datetime",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "passenger_count",
        "payment_type",
    ]
    existing = [c for c in needed_cols if c in df.columns]
    df = df[existing].copy()

    # Datetime
    if "tpep_pickup_datetime" in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"], errors="coerce"
        )

    # Filters
    if "trip_distance" in df.columns:
        df = df[df["trip_distance"] > 0]

    if "fare_amount" in df.columns:
        df = df[(df["fare_amount"] >= 0) & (df["fare_amount"] <= 300)]

    # Drop rows missing core fields
    required = [c for c in ["tpep_pickup_datetime", "trip_distance", "fare_amount"] if c in df.columns]
    df = df.dropna(subset=required)

    df = df.reset_index(drop=True)
    return df


def add_pretrip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-trip-safe features based on pickup time and distance.
    """
    df = df.copy()

    dt = df["tpep_pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour
    df["pickup_day"] = dt.dt.day
    df["pickup_weekday"] = dt.dt.weekday

    df["is_night"] = df["pickup_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)

    bins = [0, 1, 3, 7, 15, float("inf")]
    labels = ["0-1", "1-3", "3-7", "7-15", "15+"]
    df["distance_bin"] = pd.cut(
        df["trip_distance"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    return df


def build_feature_matrix(df: pd.DataFrame, feature_columns: list | None):
    """
    Create feature matrix X aligned with training-time feature_columns.
    If feature_columns is None, use current columns as-is.
    """
    df_feat = add_pretrip_features(df)

    # Base numeric features
    base_cols = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_day",
        "pickup_weekday",
        "is_night",
        "is_rush_hour",
        "is_weekend",
    ]
    base_cols = [c for c in base_cols if c in df_feat.columns]
    X = df_feat[base_cols].copy()

    # Distance bin dummies
    if "distance_bin" in df_feat.columns:
        dist_dummies = pd.get_dummies(df_feat["distance_bin"], prefix="dist")
        X = pd.concat([X, dist_dummies], axis=1)

    # Payment type dummies
    if "payment_type" in df_feat.columns:
        pay_dummies = pd.get_dummies(df_feat["payment_type"], prefix="payment")
        X = pd.concat([X, pay_dummies], axis=1)

    X = X.fillna(0.0)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
    else:
        feature_columns = list(X.columns)

    return X, feature_columns


# --------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------
def run_inference_on_dataframe(
    df_raw: pd.DataFrame,
    scaler,
    fare_model,
    tip_model,
    feature_columns: list | None,
):
    """
    Clean raw TLC data, build features, scale them, and run predictions.
    Returns (df_with_preds, metrics_dict).
    """
    df_clean = clean_taxi_data(df_raw)
    if df_clean.empty:
        return df_clean, {}

    X, feature_columns = build_feature_matrix(df_clean, feature_columns)

    X_scaled = scaler.transform(X)

    # Fare prediction
    fare_pred = fare_model.predict(X_scaled)

    # Tip prediction (sklearn logistic regression)
    tip_prob = tip_model.predict_proba(X_scaled)[:, 1]
    tip_bool = tip_prob >= 0.5

    df_out = df_clean.copy()
    df_out["pred_fare"] = fare_pred
    df_out["pred_tip_prob"] = tip_prob
    df_out["pred_tip_bool"] = tip_bool

    # Optional metrics (only if true targets exist)
    metrics = {}
    if "fare_amount" in df_clean.columns:
        metrics["fare_rmse"] = mean_squared_error(
            df_clean["fare_amount"], fare_pred, squared=False
        )
        metrics["fare_r2"] = r2_score(df_clean["fare_amount"], fare_pred)

    if "tip_amount" in df_clean.columns:
        y_tip_true = (df_clean["tip_amount"] > 0).astype(int)
        metrics["tip_acc"] = accuracy_score(y_tip_true, tip_bool)
        metrics["tip_auc"] = roc_auc_score(y_tip_true, tip_prob)

    return df_out, metrics
