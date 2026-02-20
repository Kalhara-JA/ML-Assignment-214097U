"""Train and evaluate traditional ML models for monthly arrivals forecasting."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

# Ensure plotting works in restricted environments (e.g., CI/Cloud sandboxes).
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SEED = 42
TARGET_COLUMN = "arrivals"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-safe time-series features required for model training."""
    out = df.copy()
    # Cyclical encoding preserves month seasonality better than raw month index alone.
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    # Lag/rolling features are shifted to avoid peeking into the current target month.
    out["lag_1"] = out[TARGET_COLUMN].shift(1)
    out["lag_2"] = out[TARGET_COLUMN].shift(2)
    out["lag_3"] = out[TARGET_COLUMN].shift(3)
    out["lag_12"] = out[TARGET_COLUMN].shift(12)
    out["rolling_3"] = out[TARGET_COLUMN].shift(1).rolling(3).mean()
    out["rolling_6"] = out[TARGET_COLUMN].shift(1).rolling(6).mean()
    out["pct_change_1"] = out[TARGET_COLUMN].pct_change(1).shift(1)
    out["pct_change_12"] = out[TARGET_COLUMN].pct_change(12).shift(1)
    # Percentage changes can explode when prior month is zero (pandemic months).
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics used in comparison and reporting."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    """Run model tuning, evaluation, artifact export, and plotting."""
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "sri_lanka_tourism_monthly_arrivals_2016_2025.csv"
    output_dir = root / "outputs"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = create_features(df)

    feature_cols = [
        "year",
        "month",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_3",
        "rolling_6",
        "pct_change_1",
        "pct_change_12",
    ]

    x_all = feature_df[feature_cols]
    y_all = feature_df[TARGET_COLUMN]
    date_all = feature_df["date"]

    n_total = len(feature_df)
    # Fixed holdout windows keep evaluation chronological and comparable.
    n_test = 12
    n_val = 12
    n_train = n_total - n_val - n_test
    if n_train <= 24:
        raise RuntimeError("Not enough training rows after feature generation and split.")

    # Chronological split: no shuffling for time-series data.
    x_train, y_train = x_all.iloc[:n_train], y_all.iloc[:n_train]
    x_val, y_val = x_all.iloc[n_train : n_train + n_val], y_all.iloc[n_train : n_train + n_val]
    x_test, y_test = x_all.iloc[n_train + n_val :], y_all.iloc[n_train + n_val :]
    date_test = date_all.iloc[n_train + n_val :]

    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "ridge_regression": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=SEED))]),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        ),
        "svr_rbf": (
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.01, 0.05, 0.1],
                "model__gamma": ["scale", 0.01, 0.001],
            },
        ),
        "random_forest": (
            RandomForestRegressor(random_state=SEED),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 6, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        ),
    }

    # TimeSeriesSplit respects temporal order during hyperparameter tuning.
    cv = TimeSeriesSplit(n_splits=5)
    comparison_rows: list[dict[str, Any]] = []
    best_model_name = ""
    best_model = None
    best_val_rmse = float("inf")
    best_test_pred = None

    for model_name, (estimator, param_grid) in models.items():
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        grid.fit(x_train, y_train)

        tuned_model = grid.best_estimator_
        val_pred = tuned_model.predict(x_val)
        test_pred = tuned_model.predict(x_test)
        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        row = {
            "model": model_name,
            "cv_best_neg_rmse": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        comparison_rows.append(row)

        # Select the final model using validation RMSE only (never test metrics).
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_model_name = model_name
            best_model = tuned_model
            best_test_pred = test_pred

    if best_model is None or best_test_pred is None:
        raise RuntimeError("Model selection failed.")

    comparison_df = pd.DataFrame(comparison_rows).sort_values("val_rmse")
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    test_metrics = regression_metrics(y_test, best_test_pred)
    (output_dir / "best_model_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    # Persist month-level predictions for report tables and app demos.
    predictions_df = pd.DataFrame(
        {
            "date": date_test.dt.strftime("%Y-%m-%d"),
            "actual_arrivals": y_test.values,
            "predicted_arrivals": np.round(best_test_pred).astype(int),
        }
    )
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "validation RMSE (lower is better)",
        "random_seed": SEED,
        "split_sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "feature_columns": feature_cols,
    }
    (models_dir / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump(best_model, models_dir / "best_model.joblib")

    fig_cmp, ax_cmp = plt.subplots(figsize=(7.5, 4.8))
    plot_df = comparison_df.set_index("model")[["val_rmse", "test_rmse"]]
    plot_df.plot(kind="bar", ax=ax_cmp)
    ax_cmp.set_ylabel("RMSE (arrivals)")
    ax_cmp.set_title("Model Comparison (Validation vs Test)")
    ax_cmp.grid(axis="y", linestyle="--", alpha=0.4)
    fig_cmp.tight_layout()
    fig_cmp.savefig(figures_dir / "model_comparison_rmse.png", dpi=220)
    plt.close(fig_cmp)

    fig_pred, ax_pred = plt.subplots(figsize=(9, 4.8))
    ax_pred.plot(date_test, y_test.values, marker="o", label="Actual")
    ax_pred.plot(date_test, best_test_pred, marker="s", linestyle="--", label=f"Predicted ({best_model_name})")
    ax_pred.set_title("Test Set: Actual vs Predicted Tourist Arrivals")
    ax_pred.set_xlabel("Month")
    ax_pred.set_ylabel("Arrivals")
    ax_pred.grid(alpha=0.3)
    ax_pred.legend()
    fig_pred.autofmt_xdate()
    fig_pred.tight_layout()
    fig_pred.savefig(figures_dir / "test_actual_vs_predicted.png", dpi=220)
    plt.close(fig_pred)

    print(f"Best model: {best_model_name}")
    print(f"Validation RMSE: {best_val_rmse:.2f}")
    print(f"Saved model comparison to: {output_dir / 'model_comparison.csv'}")
    print(f"Saved best model to: {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
