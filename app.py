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
import streamlit.components.v1 as components

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
APP_PUBLIC_URL = "https://lk-tourism-arrival-predictor.streamlit.app/"
STUDENT_NAME = "Kalhara J.A.K."
STUDENT_INDEX = "214097U"

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


def inject_custom_styles() -> None:
    """Apply app-wide styling for a more guided, beginner-friendly interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --brand-ink: #10324a;
            --brand-ocean: #1f6f8b;
            --brand-deep: #0f4c5c;
            --brand-accent: #f4a259;
            --paper: #ffffff;
            --paper-soft: #f6f9fc;
            --line: #d6e2ec;
            --text-main: #1b2a34;
            --text-soft: #506273;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: "Manrope", sans-serif;
            color: var(--text-main);
            background:
                radial-gradient(circle at 15% 12%, #e0eef8 0%, transparent 32%),
                radial-gradient(circle at 90% 2%, #fef3df 0%, transparent 28%),
                linear-gradient(180deg, #f7fbff 0%, #eef4f8 100%);
        }
        header[data-testid="stHeader"] {
            background: linear-gradient(180deg, #f7fbff 0%, #eef4f8 100%) !important;
            border-bottom: 1px solid #dbe6ee;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 0.7rem !important;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.7rem !important;
        }

        h1, h2, h3, h4, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.01em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12344f 0%, #1f5d74 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }
        [data-testid="stSidebar"] * {
            color: #f7fbff;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #e7f0f5;
        }
        [data-testid="stSidebar"] code {
            color: #15364c !important;
            background: #e8eef5 !important;
            border: 1px solid #d2deea;
            border-radius: 6px;
            padding: 0.08rem 0.35rem;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
            background: #f2f6fa !important;
            border-color: #20a487 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input,
        [data-testid="stSidebar"] input[type="number"] {
            color: #10324a !important;
            -webkit-text-fill-color: #10324a !important;
            caret-color: #10324a !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder {
            color: #6e7f8d !important;
            opacity: 1;
        }
        [data-testid="stSidebar"] .mini-note,
        [data-testid="stSidebar"] .mini-note * {
            color: #4f3e2b !important;
        }
        [data-testid="stSidebar"] .mini-note code {
            color: #15364c !important;
            background: #f3f7fb !important;
            border: 1px solid #d9e4ef;
        }

        .hero-wrap {
            background: linear-gradient(120deg, #0f3a57 0%, #1f6f8b 55%, #2e8d78 100%);
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            color: #f8fcff;
            box-shadow: 0 14px 30px rgba(14, 55, 80, 0.18);
            animation: fadeUp 0.5s ease-out;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(1.35rem, 2.7vw, 2.05rem);
            line-height: 1.2;
            color: #ffffff;
        }

        .hero-sub {
            margin-top: 0.55rem;
            margin-bottom: 0.75rem;
            font-size: 0.97rem;
            color: #ddedf7;
            max-width: 92ch;
        }

        .chip-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.77rem;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.12);
            color: #f8fcff;
        }

        .soft-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 14px rgba(13, 49, 71, 0.07);
            margin: 0.35rem 0 0.65rem;
            animation: fadeUp 0.45s ease-out;
        }

        .soft-card h4 {
            margin: 0 0 0.35rem;
            color: var(--brand-ink);
            font-size: 1rem;
        }

        .soft-card p {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.92rem;
        }

        .mini-note {
            background: #fef8ee;
            border-left: 4px solid var(--brand-accent);
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            margin: 0.4rem 0 0.8rem;
            color: #5b4832;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            color: var(--brand-ink);
        }

        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px;
            border: none;
            font-weight: 700;
            background: linear-gradient(110deg, #156080 0%, #2b8a78 100%);
            color: #ffffff;
            box-shadow: 0 6px 14px rgba(27, 104, 136, 0.28);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 9px 18px rgba(26, 102, 132, 0.36);
        }

        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #dce6ef;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(summary: dict, best_model_name: str) -> None:
    """Render the top hero section with context and quick facts."""
    coverage = summary["date_range"]
    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">Sri Lanka Tourism Arrival Predictor</h1>
            <p class="hero-sub">
                Forecast monthly tourist arrivals with traditional machine learning and understand feature impact using SHAP.
                This interface is designed for beginner-friendly prediction and explanation workflows.
            </p>
            <div class="chip-row">
                <span class="chip">Student: {STUDENT_NAME}</span>
                <span class="chip">Index: {STUDENT_INDEX}</span>
                <span class="chip">Best Model: {best_model_name.replace("_", " ").title()}</span>
                <span class="chip">Coverage: {coverage["start"]} to {coverage["end"]}</span>
                <span class="chip">Data Source: SLTDA</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_soft_card(title: str, description: str) -> None:
    """Render a compact informational card used throughout the app."""
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_seo_metadata(summary: dict) -> None:
    """Inject SEO metadata into the parent document head for Streamlit deployments."""
    description = (
        "Sri Lanka Tourism Arrival Predictor: forecast monthly tourist arrivals using traditional "
        "machine learning with SHAP explainability, based on SLTDA public data."
    )
    keywords = (
        "Sri Lanka tourism forecast, tourist arrivals prediction, Streamlit machine learning app, "
        "traditional ML, SHAP explainability, SLTDA dataset, Kalhara J.A.K."
    )
    coverage = f"{summary['date_range']['start']} to {summary['date_range']['end']}"

    schema_payload = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "Sri Lanka Tourism Arrival Predictor",
        "url": APP_PUBLIC_URL,
        "description": description,
        "applicationCategory": "BusinessApplication",
        "operatingSystem": "Web",
        "creator": {"@type": "Person", "name": "Kalhara J.A.K."},
        "author": {"@type": "Person", "name": "Kalhara J.A.K."},
        "about": "Monthly tourist arrival forecasting for Sri Lanka using traditional machine learning and SHAP.",
        "isAccessibleForFree": True,
        "dataset": {
            "@type": "Dataset",
            "name": "Sri Lanka Monthly Tourist Arrivals (SLTDA)",
            "description": f"Public SLTDA monthly arrivals statistics, coverage {coverage}.",
            "url": summary["source_base_url"],
        },
    }

    meta_payload = {
        "title": "Sri Lanka Tourism Arrival Predictor | Traditional ML + SHAP",
        "description": description,
        "keywords": keywords,
        "author": "Kalhara J.A.K. (214097U)",
        "canonical": APP_PUBLIC_URL,
        "og_type": "website",
        "og_url": APP_PUBLIC_URL,
        "og_title": "Sri Lanka Tourism Arrival Predictor",
        "og_description": description,
        "twitter_card": "summary_large_image",
        "twitter_title": "Sri Lanka Tourism Arrival Predictor",
        "twitter_description": description,
        "robots": "index,follow",
        "schema_json": json.dumps(schema_payload),
    }

    # Script runs inside a component iframe and updates parent <head> safely.
    components.html(
        f"""
        <script>
        (function() {{
          const head = window.parent?.document?.head;
          if (!head) return;

          const meta = {json.dumps(meta_payload)};

          function upsertMetaByName(name, content) {{
            if (!name || !content) return;
            let el = head.querySelector(`meta[name="${{name}}"]`);
            if (!el) {{
              el = window.parent.document.createElement("meta");
              el.setAttribute("name", name);
              head.appendChild(el);
            }}
            el.setAttribute("content", content);
          }}

          function upsertMetaByProperty(property, content) {{
            if (!property || !content) return;
            let el = head.querySelector(`meta[property="${{property}}"]`);
            if (!el) {{
              el = window.parent.document.createElement("meta");
              el.setAttribute("property", property);
              head.appendChild(el);
            }}
            el.setAttribute("content", content);
          }}

          function upsertCanonical(href) {{
            if (!href) return;
            let link = head.querySelector('link[rel="canonical"]');
            if (!link) {{
              link = window.parent.document.createElement("link");
              link.setAttribute("rel", "canonical");
              head.appendChild(link);
            }}
            link.setAttribute("href", href);
          }}

          function upsertJsonLd(schemaJson) {{
            if (!schemaJson) return;
            let scriptTag = head.querySelector('script[type="application/ld+json"][data-seo="lk-tourism-app"]');
            if (!scriptTag) {{
              scriptTag = window.parent.document.createElement("script");
              scriptTag.setAttribute("type", "application/ld+json");
              scriptTag.setAttribute("data-seo", "lk-tourism-app");
              head.appendChild(scriptTag);
            }}
            scriptTag.textContent = schemaJson;
          }}

          window.parent.document.title = meta.title;
          upsertMetaByName("description", meta.description);
          upsertMetaByName("keywords", meta.keywords);
          upsertMetaByName("author", meta.author);
          upsertMetaByName("robots", meta.robots);
          upsertMetaByName("twitter:card", meta.twitter_card);
          upsertMetaByName("twitter:title", meta.twitter_title);
          upsertMetaByName("twitter:description", meta.twitter_description);
          upsertMetaByProperty("og:type", meta.og_type);
          upsertMetaByProperty("og:url", meta.og_url);
          upsertMetaByProperty("og:title", meta.og_title);
          upsertMetaByProperty("og:description", meta.og_description);
          upsertCanonical(meta.canonical);
          upsertJsonLd(meta.schema_json);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


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
    st.set_page_config(
        page_title="Sri Lanka Tourism Arrival Predictor",
        page_icon="LKA",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()

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
    latest_arrivals = float(raw_df["arrivals"].iloc[-1])

    inject_seo_metadata(summary)
    render_hero(summary, best_model_name)
    st.markdown('<div class="mini-note"><strong>Quick start:</strong> Choose your forecast inputs in the left panel, click <code>Generate Forecast</code>, then use tabs to inspect predictions and SHAP explanations.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Forecast Controls")
        st.caption("Set your scenario, then generate the forecast.")
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
        st.markdown(
            """
            <div class="mini-note">
                <strong>Tip:</strong> Click <code>Generate Forecast</code> every time you change horizon or scenario.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Input Guide")
        st.write("1. `Forecast horizon`: future months to predict (1 to 12).")
        st.write("2. `Scenario`:")
        st.write("- `Baseline`: use latest real value from dataset.")
        st.write("- `Custom latest arrivals`: test what-if conditions.")
        st.write(f"Latest observed month: **{latest_date.strftime('%Y-%m')}**")
        st.write(f"Latest observed arrivals: **{with_thousands(latest_arrivals)}**")
        st.markdown("---")
        st.markdown("### Model Notes")
        st.write("- `Ridge Regression`: linear baseline.")
        st.write("- `SVR (RBF)`: kernel-based non-linear method.")
        st.write("- `Random Forest`: non-linear tree ensemble (best model).")

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
        st.subheader("Start Here")
        intro_left, intro_right = st.columns([1.15, 1], gap="large")

        with intro_left:
            render_soft_card(
                "What this app does",
                "Predicts future monthly tourist arrivals to Sri Lanka using trained traditional ML and explains prediction behavior with SHAP.",
            )
            render_soft_card(
                "How to use in 4 steps",
                "1) Set horizon and scenario in sidebar. 2) Click Generate Forecast. 3) Review the Predict tab. 4) Check Why This Prediction for explainability.",
            )
            render_soft_card(
                "What you will get",
                "Forecast table, trend chart, KPI summary, downloadable CSV, and feature-level SHAP interpretation.",
            )

        with intro_right:
            st.markdown("#### Current Model Snapshot")
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            c1.metric("Best Model", best_model_name.replace("_", " ").title())
            c2.metric("Test RMSE", with_thousands(metrics["rmse"]))
            c3.metric("Test MAE", with_thousands(metrics["mae"]))
            c4.metric("Test R2", f"{metrics['r2']:.3f}")
            st.markdown(
                """
                <div class="mini-note">
                    <strong>Meaning:</strong> Lower RMSE/MAE is better. R2 closer to 1 means better fit to real data.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Recent Historical Trend (Last 24 Months)")
        history_plot = raw_df.tail(24).set_index("date")[["arrivals"]]
        st.line_chart(history_plot)

        if PREDICTION_PLOT_PATH.exists():
            st.image(str(PREDICTION_PLOT_PATH), caption="Saved Test Plot: Actual vs Predicted (Test Split)")

        with st.expander("Model Comparison Table", expanded=False):
            show_cols = ["model", "val_rmse", "test_rmse", "test_mae", "test_r2"]
            display_df = comparison_df[show_cols].copy()
            display_df["model"] = display_df["model"].str.replace("_", " ").str.title()
            st.dataframe(display_df, use_container_width=True)

        with st.expander("Public Source Details", expanded=False):
            st.write(f"Source owner: **{summary['source_owner']}**")
            st.write(f"Collection: **{summary['source_collection']}**")
            st.write(f"Base URL: {summary['source_base_url']}")
            st.write("Year-specific source pages:")
            for year, url in summary["source_urls"].items():
                st.write(f"- {year}: {url}")

    with tab2:
        st.subheader("Predict")
        p1, p2 = st.columns([1.4, 1], gap="large")

        with p1:
            render_soft_card(
                "Active Forecast Setup",
                f"Scenario: {scenario_label}. Horizon: {used_horizon} month(s). Change values in sidebar and click Generate Forecast to refresh.",
            )
        with p2:
            st.markdown(
                f"""
                <div class="mini-note">
                    <strong>Latest baseline point:</strong> {latest_date.strftime('%Y-%m')} with {with_thousands(latest_arrivals)} arrivals.
                </div>
                """,
                unsafe_allow_html=True,
            )

        first_pred = int(forecast_df["predicted_arrivals"].iloc[0])
        total_pred = int(forecast_df["predicted_arrivals"].sum())
        avg_pred = int(forecast_df["predicted_arrivals"].mean())
        peak_row = forecast_df.loc[forecast_df["predicted_arrivals"].idxmax()]

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("First forecast month", forecast_df.iloc[0]["date"][:7])
        f2.metric("First predicted arrivals", with_thousands(first_pred))
        f3.metric("Average monthly prediction", with_thousands(avg_pred))
        f4.metric("Total predicted arrivals", with_thousands(total_pred))
        st.caption(f"Peak month in forecast: {peak_row['date'][:7]} ({with_thousands(peak_row['predicted_arrivals'])})")

        st.markdown("#### Output Table Meaning")
        st.write("- `date`: forecast month.")
        st.write("- `predicted_arrivals`: predicted arrivals for that month.")

        forecast_table = forecast_df[["date", "predicted_arrivals"]].copy()
        st.dataframe(forecast_table, use_container_width=True)

        st.markdown("#### Historical + Forecast Chart")
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
        st.subheader("Why This Prediction")
        top_shap = shap_df.sort_values("mean_abs_shap", ascending=False).head(10)
        explain_table = build_feature_explain_table(top_shap)
        top_features = top_shap["feature"].head(5).tolist()

        x1, x2 = st.columns([1.1, 1], gap="large")

        with x1:
            render_soft_card(
                "Explainability method used",
                "SHAP (SHapley Additive exPlanations) quantifies how strongly each feature affects predictions.",
            )
            render_soft_card(
                "What the model has learned",
                "The model learns short-term momentum and yearly seasonality. Recent month behavior and same-month-last-year signals are dominant.",
            )
            render_soft_card(
                "Domain alignment",
                "Behavior matches tourism demand patterns: lag-based momentum and seasonal cycles are expected to be influential.",
            )
            st.markdown("**Most influential features**")
            st.write(", ".join(f"`{name}`" for name in top_features))

        with x2:
            st.markdown("**Global SHAP Importance (Top 10)**")
            st.bar_chart(top_shap.set_index("feature")["mean_abs_shap"])
            st.caption("Higher mean |SHAP| means stronger overall influence on model output.")

        st.markdown("#### Feature-by-Feature Explanation")
        st.dataframe(explain_table, use_container_width=True)

        c_left, c_right = st.columns(2)
        c_left.image(str(SHAP_BEESWARM_PATH), caption="SHAP Summary (Beeswarm)")
        c_right.image(str(SHAP_BAR_PATH), caption="SHAP Global Importance (Bar)")

        st.success("Interpretation: higher-ranked SHAP parameters have stronger influence on predictions.")

        with st.expander("Recording Checklist", expanded=False):
            st.write("1. Show model metrics and source details in Start Here.")
            st.write("2. Change scenario and horizon in sidebar, then click Generate Forecast.")
            st.write("3. Show forecast table/chart and download CSV.")
            st.write("4. Open Why This Prediction tab and explain top features.")

    with tab4:
        st.subheader("About Sri Lanka Tourism Arrival Predictor")
        a1, a2 = st.columns([1.2, 1], gap="large")
        with a1:
            render_soft_card(
                "Purpose",
                "This app predicts future monthly tourist arrivals to Sri Lanka using traditional machine learning and provides SHAP-based explainability.",
            )
            render_soft_card(
                "How to interpret outputs",
                "Use Predict tab for forecast values and trend shape, then use Why This Prediction to identify the strongest influencing parameters.",
            )
        with a2:
            st.markdown(
                f"""
                <div class="mini-note">
                    <strong>Dataset coverage:</strong> {summary['date_range']['start']} to {summary['date_range']['end']}<br/>
                    <strong>Rows:</strong> {summary['n_rows']} monthly records
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Model Performance (Test Set)")
        st.dataframe(model_performance_table(comparison_df), use_container_width=True, hide_index=True)

        st.markdown("#### What These Metrics Mean")
        metric_help_df = pd.DataFrame(
            [
                {"metric": "RMSE", "description": METRIC_DESCRIPTIONS["RMSE"]},
                {"metric": "MAE", "description": METRIC_DESCRIPTIONS["MAE"]},
                {"metric": "R2", "description": METRIC_DESCRIPTIONS["R2"]},
                {"metric": "SHAP (mean |value|)", "description": METRIC_DESCRIPTIONS["SHAP (mean |value|)"]},
            ]
        )
        st.dataframe(metric_help_df, use_container_width=True, hide_index=True)

        st.markdown("#### Top SHAP Features (Global Importance)")
        st.dataframe(shap_about_table(shap_df), use_container_width=True, hide_index=True)

        st.markdown("#### How Inputs Work")
        st.write("- Required: `Forecast horizon` and `Scenario` selection.")
        st.write("- Optional: custom latest-arrivals value for what-if simulation.")
        st.write("- `Baseline` uses actual latest observed data from the dataset.")

        st.markdown("#### How to Read Outputs")
        st.write("- `Predict` tab table gives month-by-month forecast values.")
        st.write("- KPI cards summarize first-month, average, total, and peak forecast levels.")
        st.write("- `Why This Prediction` tab explains feature influence using SHAP.")

        st.markdown("#### Data Source")
        st.write(f"- Owner: {summary['source_owner']}")
        st.write(f"- Collection: {summary['source_collection']}")
        st.write(f"- Coverage: {summary['date_range']['start']} to {summary['date_range']['end']}")
        st.write(f"- Source URL: {summary['source_base_url']}")

        with st.expander("Year-Specific Source Pages", expanded=False):
            for year, url in summary["source_urls"].items():
                st.write(f"- {year}: {url}")

        st.markdown("#### Important Disclaimer")
        st.write("- This is a decision-support forecasting tool, not a guaranteed future value.")
        st.write("- Extreme events or policy shifts may change real outcomes beyond historical patterns.")

        st.markdown("#### Limitations (Short)")
        st.write("- Dataset is relatively small for time-series ML (monthly records only).")
        st.write("- COVID-period structural breaks can reduce pattern stability.")
        st.write("- Model currently uses limited external drivers beyond historical arrivals.")

        st.markdown("#### Future Improvements (Short)")
        st.write("- Add external features such as flight capacity and macroeconomic indicators.")
        st.write("- Retrain model regularly as new monthly data is released.")
        st.write("- Add uncertainty intervals and model monitoring in production.")


if __name__ == "__main__":
    main()
