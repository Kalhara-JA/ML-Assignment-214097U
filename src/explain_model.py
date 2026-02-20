"""Generate SHAP-based explainability artifacts for the trained best model."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib

# Ensure matplotlib cache paths are writable across local/cloud environments.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_and_evaluate import TARGET_COLUMN, create_features


def main() -> None:
    """Compute SHAP values and export explainability tables/figures."""
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "sri_lanka_tourism_monthly_arrivals_2016_2025.csv"
    output_dir = root / "outputs"
    explain_dir = output_dir / "explainability"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    explain_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = create_features(df)

    metadata = json.loads((models_dir / "best_model_metadata.json").read_text(encoding="utf-8"))
    feature_cols = metadata["feature_columns"]
    model_name = metadata["best_model_name"]

    x_all = feature_df[feature_cols]
    y_all = feature_df[TARGET_COLUMN]

    n_total = len(feature_df)
    # Use the same chronological split strategy as training script.
    n_test = 12
    n_val = 12
    n_train = n_total - n_val - n_test

    x_train = x_all.iloc[:n_train]
    x_test = x_all.iloc[n_train + n_val :]

    model = joblib.load(models_dir / "best_model.joblib")

    final_estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_estimator = model.named_steps["model"]

    # Prefer TreeExplainer to avoid numba-heavy generic explainers and keep SHAP stable.
    if hasattr(final_estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(final_estimator)
        shap_values_raw = explainer.shap_values(x_test)
        if isinstance(shap_values_raw, list):
            shap_values_array = np.asarray(shap_values_raw[0], dtype=float)
        else:
            shap_values_array = np.asarray(shap_values_raw, dtype=float)
    else:
        # Fallback path for non-tree estimators.
        background = x_train.sample(n=min(30, len(x_train)), random_state=42).to_numpy()
        predict_fn = lambda data: model.predict(pd.DataFrame(data, columns=feature_cols))  # noqa: E731
        kernel = shap.KernelExplainer(predict_fn, background)
        shap_values_raw = kernel.shap_values(x_test.to_numpy(), nsamples=100)
        if isinstance(shap_values_raw, list):
            shap_values_array = np.asarray(shap_values_raw[0], dtype=float)
        else:
            shap_values_array = np.asarray(shap_values_raw, dtype=float)

    # Global feature importance from SHAP = mean absolute SHAP value per feature.
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    shap_df.to_csv(explain_dir / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 5.8))
    shap.summary_plot(
        shap_values_array,
        features=x_test.to_numpy(),
        feature_names=feature_cols,
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Summary (Beeswarm) - {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.5, 5.5))
    shap.summary_plot(
        shap_values_array,
        features=x_test.to_numpy(),
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Global Importance (Bar) - {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_bar.png", dpi=220, bbox_inches="tight")
    plt.close()

    summary = {
        "best_model": model_name,
        "xai_method": "SHAP",
        "top_shap_features": shap_df["feature"].head(5).tolist(),
    }
    (explain_dir / "explainability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved SHAP importance to: {explain_dir / 'shap_feature_importance.csv'}")
    print(f"Saved explainability summary to: {explain_dir / 'explainability_summary.json'}")


if __name__ == "__main__":
    main()
