"""Baseline models for time-series forecasting"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Dict


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "sales"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple baseline models on validation data.

    Baselines are simple forecasting methods that serve as a sanity check.
    Any ML model should outperform these baselines to be considered useful.

    Baselines evaluated:
    1. Global Mean: Use training set average as prediction
    2. Lag-1 (Naive): Use yesterday's sales as prediction
    3. Rolling Mean 7-day: Use 7-day rolling average

    Args:
        train_df: Training data with 'sales' column
        val_df: Validation data with lag features already computed
        target_col: Target column name (default: "sales")

    Returns:
        Dictionary with baseline names as keys and metrics as values:
        {
            "Global Mean": {"MAE": 18.45, "RMSE": 23.12},
            "Lag-1": {"MAE": 4.12, "RMSE": 5.89},
            ...
        }

    Example:
        >>> results = evaluate_baselines(train_df, val_df)
        >>> for name, metrics in results.items():
        ...     print(f"{name}: MAE={metrics['MAE']:.2f}")
    """
    # Filter out rows with missing lag features
    val_clean = val_df.dropna(subset=["sales_lag_1", "roll_mean_7"])
    y_true = val_clean[target_col]

    results = {}

    # Baseline 1: Global mean from training set
    global_mean = train_df[target_col].mean()
    y_pred_mean = [global_mean] * len(val_clean)
    results["Global Mean"] = {
        "MAE": mean_absolute_error(y_true, y_pred_mean),
        "RMSE": root_mean_squared_error(y_true, y_pred_mean),
    }

    # Baseline 2: Lag-1 (yesterday's sales)
    y_pred_lag1 = val_clean["sales_lag_1"]
    results["Lag-1 (yesterday)"] = {
        "MAE": mean_absolute_error(y_true, y_pred_lag1),
        "RMSE": root_mean_squared_error(y_true, y_pred_lag1),
    }

    # Baseline 3: 7-day rolling mean
    y_pred_roll7 = val_clean["roll_mean_7"]
    results["Rolling Mean 7-day"] = {
        "MAE": mean_absolute_error(y_true, y_pred_roll7),
        "RMSE": root_mean_squared_error(y_true, y_pred_roll7),
    }

    # Print results table
    print("=== Baseline Results (Validation Set) ===")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10}")
    print("-" * 47)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f}")
    print()
    print(f"Validation samples: {len(val_clean):,}")

    return results
