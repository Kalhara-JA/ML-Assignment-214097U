"""Streamlit front-end for Sri Lanka Tourism Arrival Predictor.

The app provides:
- beginner-friendly forecast controls and output views
- first-run artifact bootstrapping for Streamlit Cloud
- SHAP-based explanation views for model interpretation
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


DATA_PATH = ROOT / "data" / "raw" / "sri_lanka_tourism_monthly_arrivals_2016_2025.csv"
METADATA_PATH = ROOT / "outputs" / "models" / "best_model_metadata.json"
MODEL_PATH = ROOT / "outputs" / "models" / "best_model.joblib"
SUMMARY_PATH = ROOT / "outputs" / "step1_dataset_summary.json"
COMPARISON_PATH = ROOT / "outputs" / "model_comparison.csv"
METRICS_PATH = ROOT / "outputs" / "best_model_metrics.json"
SHAP_IMPORTANCE_PATH = ROOT / "outputs" / "explainability" / "shap_feature_importance.csv"
PREDICTION_PLOT_PATH = ROOT / "outputs" / "figures" / "test_actual_vs_predicted.png"
SHAP_BEESWARM_PATH = ROOT / "outputs" / "figures" / "shap_summary_beeswarm.png"
SHAP_BAR_PATH = ROOT / "outputs" / "figures" / "shap_summary_bar.png"

FEATURE_DESCRIPTIONS = {
    "year": "Calendar year of the month being predicted.",
    "month": "Month number (1=Jan, 12=Dec).",
    "month_sin": "Seasonality signal for the month (cyclical encoding).",
    "month_cos": "Seasonality signal for the month (cyclical encoding).",
    "lag_1": "Arrivals in the previous month.",
    "lag_2": "Arrivals two months before.",
    "lag_3": "Arrivals three months before.",
    "lag_12": "Arrivals in the same month last year.",
    "rolling_3": "Average arrivals over last 3 months.",
    "rolling_6": "Average arrivals over last 6 months.",
    "pct_change_1": "Recent month-to-month growth rate.",
    "pct_change_12": "Year-over-year growth rate trend.",
}

FEATURE_WHY_MATTERS = {
    "lag_1": "Captures short-term momentum from the previous month.",
    "lag_12": "Captures yearly seasonal pattern from the same month last year.",
    "rolling_3": "Smooths recent fluctuations using a short moving average.",
    "rolling_6": "Captures medium-term trend level.",
    "pct_change_1": "Represents recent month-to-month growth or decline.",
    "pct_change_12": "Represents year-over-year direction and intensity.",
    "month": "Models monthly seasonality differences.",
    "month_sin": "Seasonality encoding for cyclic month behavior.",
    "month_cos": "Seasonality encoding for cyclic month behavior.",
    "year": "Captures long-term drift over years.",
    "lag_2": "Adds extra short-term memory from two months back.",
    "lag_3": "Adds extra short-term memory from three months back.",
}

METRIC_DESCRIPTIONS = {
    "RMSE": "Root Mean Squared Error. Penalizes larger errors more strongly. Lower is better.",
    "MAE": "Mean Absolute Error. Average absolute prediction error. Lower is better.",
    "R2": "Coefficient of determination. Proportion of variance explained by the model. Closer to 1 is better.",
    "SHAP (mean |value|)": "Average absolute SHAP contribution of a feature. Higher means stronger influence on predictions.",
}


def safe_pct_change(new_value: float, old_value: float) -> float:
    """Return percentage change while safely handling zero denominator."""
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def build_feature_row(history_values: list[float], target_date: pd.Timestamp) -> dict[str, float]:
    """Construct one model input row for a target month from historical values."""
    month = int(target_date.month)
    # Recreate the exact feature schema used during model training.
    return {
        "year": float(target_date.year),
        "month": float(month),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "lag_1": float(history_values[-1]),
        "lag_2": float(history_values[-2]),
        "lag_3": float(history_values[-3]),
        "lag_12": float(history_values[-12]),
        "rolling_3": float(np.mean(history_values[-3:])),
        "rolling_6": float(np.mean(history_values[-6:])),
        "pct_change_1": float(safe_pct_change(history_values[-1], history_values[-2])),
        "pct_change_12": float(safe_pct_change(history_values[-1], history_values[-13])),
    }


def with_thousands(value: float | int) -> str:
    """Format numeric values with comma separators for UI display."""
    return f"{int(round(value)):,}"


def build_feature_explain_table(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Attach plain-language parameter explanations to SHAP feature table."""
    out = shap_df.copy()
    out["what_this_parameter_means"] = out["feature"].map(FEATURE_DESCRIPTIONS).fillna("Parameter used by model.")
    out["importance_note"] = "Higher SHAP value means stronger effect on prediction."
    return out


def model_performance_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact model comparison table for about/overview sections."""
    out = comparison_df[["model", "test_rmse", "test_mae", "test_r2"]].copy()
    out["model"] = out["model"].str.replace("_", " ").str.title()
    out = out.sort_values("test_rmse")
    return out


def shap_about_table(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Build an about-page SHAP table with rank and short interpretation."""
    top = shap_df.sort_values("mean_abs_shap", ascending=False).head(8).copy().reset_index(drop=True)
    top["rank"] = top.index + 1
    top["why_it_matters"] = top["feature"].map(FEATURE_WHY_MATTERS).fillna("Contributes to model prediction.")
    top["mean_abs_shap"] = top["mean_abs_shap"].round(2)
    return top[["rank", "feature", "mean_abs_shap", "why_it_matters"]]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and return raw monthly arrivals data."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data
def load_summary() -> dict:
    """Load dataset summary metadata from JSON artifact."""
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_metadata() -> dict:
    """Load training metadata (best model + feature list)."""
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_comparison() -> pd.DataFrame:
    """Load model comparison metrics table."""
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_metrics() -> dict:
    """Load best-model test metrics used for dashboard cards."""
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    """Load precomputed SHAP global importance scores."""
    return pd.read_csv(SHAP_IMPORTANCE_PATH)


@st.cache_resource
def load_model():
    """Load the persisted trained model artifact."""
    return joblib.load(MODEL_PATH)


def forecast_arrivals(
    model,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    override_last_value: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce recursive multi-step forecasts using the trained model."""
    history = history_df[["date", "arrivals"]].copy()
    history["arrivals"] = history["arrivals"].astype(float)

    if override_last_value is not None:
        history.loc[history.index[-1], "arrivals"] = float(override_last_value)

    all_dates = list(history["date"])
    all_values = list(history["arrivals"])

    forecast_rows: list[dict[str, float | str]] = []

    # Recursive forecasting: each predicted month is fed back as future lag input.
    for idx in range(horizon):
        next_date = all_dates[-1] + pd.DateOffset(months=1)
        feature_row = build_feature_row(all_values, next_date)
        x_one = pd.DataFrame([feature_row])[feature_cols]
        raw_pred = float(model.predict(x_one)[0])
        raw_pred = max(0.0, raw_pred)
        rounded_pred = int(round(raw_pred))

        forecast_rows.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_arrivals": rounded_pred,
                **feature_row,
            }
        )

        all_dates.append(next_date)
        all_values.append(raw_pred)
        if progress_callback:
            progress_callback(idx + 1, horizon)

    return pd.DataFrame(forecast_rows), history


def check_required_files() -> list[Path]:
    """Return missing artifact paths required by the app."""
    required = [
        DATA_PATH,
        METADATA_PATH,
        MODEL_PATH,
        SUMMARY_PATH,
        COMPARISON_PATH,
        METRICS_PATH,
        SHAP_IMPORTANCE_PATH,
        SHAP_BEESWARM_PATH,
        SHAP_BAR_PATH,
    ]
    return [p for p in required if not p.exists()]


def build_required_artifacts() -> tuple[bool, str]:
    """Build dataset/model/explainability artifacts when missing on first run."""
    commands = [
        [sys.executable, "src/step1_prepare_dataset.py"],
        [sys.executable, "src/train_and_evaluate.py"],
        [sys.executable, "src/explain_model.py"],
    ]
    logs: list[str] = []

    for cmd in commands:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        logs.append(f"$ {' '.join(cmd)}\n{(proc.stdout or '').strip()}\n{(proc.stderr or '').strip()}".strip())
        if proc.returncode != 0:
            return False, "\n\n".join(logs)

    return True, "\n\n".join(logs)


def main() -> None:
    """Render and run the Streamlit application."""
    st.set_page_config(page_title="Sri Lanka Tourism Arrival Predictor", page_icon="LKA", layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .small-note { color: #6b7280; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Sri Lanka Tourism Arrival Predictor")

    missing_files = check_required_files()
    if missing_files:
        st.warning("First run setup: required artifacts are not present yet.")
        with st.expander("Missing files", expanded=False):
            for path in missing_files:
                st.write(f"- `{path}`")

        if "artifact_build_failed" not in st.session_state:
            with st.spinner("Preparing dataset, training model, and generating SHAP outputs..."):
                ok, log_text = build_required_artifacts()
            if ok:
                st.success("Setup complete. Reloading app...")
                st.rerun()
            st.session_state["artifact_build_failed"] = True
            st.session_state["artifact_build_log"] = log_text

        st.error("Automatic setup failed. Please use the run commands below and retry.")
        st.code(
            ". .venv/bin/activate\n"
            "python src/step1_prepare_dataset.py\n"
            "python src/train_and_evaluate.py\n"
            "python src/explain_model.py"
        )
        with st.expander("Build logs", expanded=False):
            st.code(st.session_state.get("artifact_build_log", "No logs captured."))
        st.stop()

    summary = load_summary()
    metadata = load_metadata()
    metrics = load_metrics()
    model = load_model()
    raw_df = load_data()
    comparison_df = load_comparison()
    shap_df = load_shap_importance()

    if len(raw_df) < 14:
        st.error("Dataset has insufficient history for lag-based forecasting. Minimum required rows: 14.")
        st.stop()

    feature_cols = metadata["feature_columns"]
    best_model_name = metadata["best_model_name"]
    latest_date = raw_df["date"].max()

    with st.sidebar:
        st.header("Inputs")
        horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)
        scenario_mode = st.radio("Scenario", options=["Baseline", "Custom latest arrivals"], index=0)
        override_val = None
        if scenario_mode == "Custom latest arrivals":
            override_val = st.number_input(
                "Set latest arrivals value",
                min_value=0.0,
                value=float(raw_df["arrivals"].iloc[-1]),
                step=1000.0,
                help="Useful for what-if analysis before generating forecasts.",
            )
        run_forecast = st.button("Generate Forecast", use_container_width=True)
        st.markdown('<p class="small-note">Tip: Click Generate Forecast after changing settings.</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**What should you input?**")
        st.write("1. `Forecast horizon`: how many future months to predict (1-12).")
        st.write("2. `Scenario`:")
        st.write("- `Baseline`: uses latest real observed value from the dataset.")
        st.write("- `Custom latest arrivals`: enter your own latest value for what-if testing.")
        st.write(f"Latest observed month: **{latest_date.strftime('%Y-%m')}**")
        st.write(f"Latest observed arrivals: **{with_thousands(raw_df['arrivals'].iloc[-1])}**")

    # Session state keeps the latest forecast visible across Streamlit reruns.
    if run_forecast or "forecast_df" not in st.session_state:
        progress_bar = None
        status_box = None

        if run_forecast:
            status_box = st.empty()
            status_box.info("Generating forecast...")
            progress_bar = st.progress(0, text="Starting forecast...")

        def on_progress(step: int, total: int) -> None:
            if progress_bar is not None:
                percent = int((step / total) * 100)
                progress_bar.progress(percent, text=f"Generating month {step} of {total}...")

        with st.spinner("Running model and generating predictions..."):
            forecast_df, history_df = forecast_arrivals(
                model=model,
                history_df=raw_df,
                feature_cols=feature_cols,
                horizon=horizon,
                override_last_value=override_val,
                progress_callback=on_progress,
            )

        if progress_bar is not None:
            progress_bar.progress(100, text="Forecast generated.")
        if status_box is not None:
            status_box.success("Forecast complete.")

        st.session_state["forecast_df"] = forecast_df
        st.session_state["history_df"] = history_df
        st.session_state["scenario_label"] = scenario_mode
        st.session_state["horizon"] = horizon

    forecast_df = st.session_state["forecast_df"]
    history_df = st.session_state["history_df"]
    scenario_label = st.session_state["scenario_label"]
    used_horizon = st.session_state["horizon"]

    tab1, tab2, tab3, tab4 = st.tabs(["Start Here", "Predict", "Why This Prediction", "About This App"])

    with tab1:
        st.subheader("How to use this app")
        st.write("1. Choose input values in the left sidebar.")
        st.write("2. Click `Generate Forecast`.")
        st.write("3. Open `Predict` tab to see outputs.")
        st.write("4. Open `Why This Prediction` to understand feature influence.")

        st.subheader("What you will get")
        st.write("- Future monthly arrival predictions")
        st.write("- A chart of historical vs predicted values")
        st.write("- SHAP explainability (which parameters influenced predictions most)")

        st.subheader("Models compared (short notes)")
        st.write("- `Ridge Regression`: linear baseline model.")
        st.write("- `SVR (RBF)`: non-linear model using support vectors and kernel mapping.")
        st.write("- `Random Forest Regressor`: tree-ensemble model that captures non-linear patterns.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", best_model_name.replace("_", " ").title())
        c2.metric("Test RMSE", with_thousands(metrics["rmse"]))
        c3.metric("Test MAE", with_thousands(metrics["mae"]))
        c4.metric("Test R2", f"{metrics['r2']:.3f}")

        with st.expander("Public Source Details", expanded=False):
            st.write(f"Source owner: **{summary['source_owner']}**")
            st.write(f"Collection: **{summary['source_collection']}**")
            st.write(f"Base URL: {summary['source_base_url']}")
            st.write("Year-specific source pages:")
            for year, url in summary["source_urls"].items():
                st.write(f"- {year}: {url}")

        with st.expander("Model Comparison (Advanced)", expanded=False):
            show_cols = ["model", "val_rmse", "test_rmse", "test_mae", "test_r2"]
            display_df = comparison_df[show_cols].copy()
            display_df["model"] = display_df["model"].str.replace("_", " ").str.title()
            st.dataframe(display_df, use_container_width=True)

        st.subheader("Recent historical trend")
        history_plot = raw_df.tail(24).set_index("date")[["arrivals"]]
        st.line_chart(history_plot)

        if PREDICTION_PLOT_PATH.exists():
            st.image(str(PREDICTION_PLOT_PATH), caption="Saved Test Plot: Actual vs Predicted")

    with tab2:
        st.subheader("Prediction Output")
        st.write(f"Scenario: **{scenario_label}**")
        st.write(f"Horizon used: **{used_horizon} months**")

        first_pred = int(forecast_df["predicted_arrivals"].iloc[0])
        total_pred = int(forecast_df["predicted_arrivals"].sum())
        avg_pred = int(forecast_df["predicted_arrivals"].mean())
        peak_row = forecast_df.loc[forecast_df["predicted_arrivals"].idxmax()]

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("First predicted month", forecast_df.iloc[0]["date"][:7])
        f2.metric("First predicted arrivals", with_thousands(first_pred))
        f3.metric("Average monthly prediction", with_thousands(avg_pred))
        f4.metric("Total predicted arrivals", with_thousands(total_pred))
        st.caption(f"Peak month in forecast: {peak_row['date'][:7]} ({with_thousands(peak_row['predicted_arrivals'])})")

        st.markdown("**Output table meaning**")
        st.write("- `date`: forecasted month")
        st.write("- `predicted_arrivals`: model's predicted tourist arrivals for that month")

        forecast_table = forecast_df[["date", "predicted_arrivals"]].copy()
        st.dataframe(forecast_table, use_container_width=True)

        hist_plot = history_df.tail(24).copy()
        hist_plot = hist_plot[["date", "arrivals"]].rename(columns={"arrivals": "historical_arrivals"})
        fc_plot = forecast_df[["date", "predicted_arrivals"]].copy()
        fc_plot["date"] = pd.to_datetime(fc_plot["date"])
        fc_plot = fc_plot.rename(columns={"predicted_arrivals": "forecast_arrivals"})
        merged_plot = pd.merge(hist_plot, fc_plot, on="date", how="outer").sort_values("date").set_index("date")
        st.line_chart(merged_plot)

        st.download_button(
            "Download Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="streamlit_forecast_output.csv",
            mime="text/csv",
        )

        with st.expander("Technical: Model Input Parameters Used for First Forecasted Month", expanded=False):
            st.dataframe(forecast_df.head(1)[["date"] + feature_cols], use_container_width=True)

    with tab3:
        st.subheader("Explainability (SHAP)")
        top_shap = shap_df.sort_values("mean_abs_shap", ascending=False).head(10)
        explain_table = build_feature_explain_table(top_shap)
        top_features = top_shap["feature"].head(5).tolist()

        st.markdown("**Method used**: SHAP (SHapley Additive exPlanations)")
        st.caption("SHAP shows how strongly each parameter influences the model's prediction.")

        st.markdown("**What the model has learned**")
        st.write(
            "The model mainly learns temporal dependency and seasonality patterns. "
            "Recent months and same-month-last-year behavior have the largest effect on next-month arrivals."
        )

        st.markdown("**Most influential features**")
        st.write(", ".join(f"`{name}`" for name in top_features))
        st.dataframe(explain_table, use_container_width=True)
        st.bar_chart(top_shap.set_index("feature")["mean_abs_shap"])

        st.markdown("**Alignment with domain knowledge**")
        st.write(
            "The behavior is consistent with tourism demand dynamics: arrivals usually follow short-term momentum "
            "(`lag_1`) and yearly seasonality (`lag_12`, month-related features)."
        )

        c_left, c_right = st.columns(2)
        c_left.image(str(SHAP_BEESWARM_PATH), caption="SHAP Summary (Beeswarm)")
        c_right.image(str(SHAP_BAR_PATH), caption="SHAP Global Importance (Bar)")

        st.success("Short interpretation: higher-ranked parameters have stronger influence on prediction.")

        with st.expander("Recording Checklist", expanded=False):
            st.write("1. Show model metrics and source details in Start Here.")
            st.write("2. Change scenario and horizon in sidebar, then click Generate Forecast.")
            st.write("3. Show forecast table/chart and download CSV.")
            st.write("4. Open Why This Prediction tab and explain top features.")

    with tab4:
        st.subheader("About Sri Lanka Tourism Arrival Predictor")
        st.write(
            "This app predicts future monthly tourist arrivals to Sri Lanka using a trained traditional "
            "machine learning model and explains model behavior with SHAP."
        )

        st.markdown("**What this app does**")
        st.write(
            "It provides quick monthly tourism-arrival forecasts and a transparent explanation of what drives each prediction."
        )

        st.markdown("**Model performance (test set)**")
        st.dataframe(model_performance_table(comparison_df), use_container_width=True, hide_index=True)

        st.markdown("**What these metrics mean**")
        metric_help_df = pd.DataFrame(
            [
                {"metric": "RMSE", "description": METRIC_DESCRIPTIONS["RMSE"]},
                {"metric": "MAE", "description": METRIC_DESCRIPTIONS["MAE"]},
                {"metric": "R2", "description": METRIC_DESCRIPTIONS["R2"]},
                {"metric": "SHAP (mean |value|)", "description": METRIC_DESCRIPTIONS["SHAP (mean |value|)"]},
            ]
        )
        st.dataframe(metric_help_df, use_container_width=True, hide_index=True)

        st.markdown("**Top SHAP features (global importance)**")
        st.dataframe(shap_about_table(shap_df), use_container_width=True, hide_index=True)

        st.markdown("**How inputs work**")
        st.write("- Required: `Forecast horizon` and `Scenario` selection.")
        st.write("- Optional: custom latest-arrivals value for what-if simulation.")
        st.write("- `Baseline` uses actual latest observed data from the dataset.")

        st.markdown("**How to read outputs**")
        st.write("- `Predict` tab table gives month-by-month forecast values.")
        st.write("- KPI cards summarize first-month, average, total, and peak forecast levels.")
        st.write("- `Why This Prediction` tab explains feature influence using SHAP.")

        st.markdown("**Data source**")
        st.write(f"- Owner: {summary['source_owner']}")
        st.write(f"- Collection: {summary['source_collection']}")
        st.write(f"- Coverage: {summary['date_range']['start']} to {summary['date_range']['end']}")
        st.write(f"- Source URL: {summary['source_base_url']}")

        st.markdown("**Important disclaimer**")
        st.write("- This is a decision-support forecasting tool, not a guaranteed future value.")
        st.write("- Extreme events or policy shifts may change real outcomes beyond historical patterns.")


if __name__ == "__main__":
    main()
