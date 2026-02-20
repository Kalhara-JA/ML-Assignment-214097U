# Demo Video Script (3-5 Minutes, Final)

## Presenter Details
- Name: `Kalhara J.A.K.`
- Index: `214097U`
- Module work: `ML Assignment`
- App URL: `https://lk-tourism-arrival-predictor.streamlit.app/`

## 0:00-0:25 Intro
### On screen
- Open the Streamlit app home view.

### Narration
Hello, I am Kalhara J.A.K., index number 214097U.  
This is my machine learning assignment on forecasting monthly tourist arrivals to Sri Lanka using traditional machine learning and SHAP explainability.

## 0:25-1:00 Problem and Dataset
### On screen
- Go to `Start Here`.
- Open `Public Source Details`.

### Narration
The problem is short-term tourism demand forecasting for Sri Lanka.  
I used publicly available monthly arrival statistics from the Sri Lanka Tourism Development Authority, or SLTDA.  
The dataset covers January 2016 to December 2025, with 120 monthly records.  
The source links are shown in the app for transparency and traceability.

## 1:00-1:45 Method and Models
### On screen
- Stay on `Start Here`.
- Open `Model Comparison Table`.

### Narration
I framed this as a supervised regression problem where the target variable is monthly arrivals.  
I engineered time-series features including lag features such as `lag_1` and `lag_12`, rolling averages such as `rolling_3` and `rolling_6`, and seasonal encodings `month_sin` and `month_cos`.  
I compared three traditional models taught in class:
- Ridge Regression
- SVR with RBF kernel
- Random Forest Regressor

I used chronological train, validation, and test splitting to avoid temporal leakage, and `TimeSeriesSplit` for model tuning.

## 1:45-2:20 Results Summary
### On screen
- Point to model metric cards and model comparison table.

### Narration
The best model is Random Forest.  
On the test set, it achieved:
- RMSE: 19,255.20
- MAE: 15,864.72
- R2: 0.842

This outperformed Ridge and SVR for this dataset.

## 2:20-3:40 Live App Walkthrough
### On screen and narration
1. In the sidebar, set forecast horizon to `6` months and keep scenario as `Baseline`.
   Narration: First I run the baseline forecast using the latest observed value from the dataset.

2. Click `Generate Forecast`.
   Narration: The app shows a loading progress and then generates predictions.

3. Open `Predict`.
   Narration: Here we get KPI cards, the monthly forecast table, and a historical-plus-forecast trend chart.

4. Click `Download Forecast CSV`.
   Narration: The forecast can be exported as CSV for reporting or further analysis.

5. Change scenario to `Custom latest arrivals`, enter a different latest value, and click `Generate Forecast` again.
   Narration: This is scenario analysis. By overriding the latest value, we can test how future predictions shift under what-if conditions.

## 3:40-4:25 Explainability
### On screen
- Open `Why This Prediction`.

### Narration
For explainability, I used SHAP.  
SHAP quantifies feature influence on predictions.

Top global features are:
1. `lag_1`
2. `lag_12`
3. `rolling_3`
4. `pct_change_12`
5. `month`

This indicates the model mainly learns short-term momentum and yearly seasonality, which is consistent with tourism domain behavior.

## 4:25-4:50 About and Metric Definitions
### On screen
- Open `About This App`.
- Point to the metric definition table.

### Narration
This section explains RMSE, MAE, R2, and SHAP importance in simple terms, so non-technical users can understand model quality and interpretation.

## 4:50-5:00 Critical Discussion and Closing
### Narration
Main limitations are small dataset size, structural breaks during COVID years, and limited external variables.  
Future improvements include adding macroeconomic and aviation indicators, periodic retraining, and uncertainty intervals.  
In summary, this project satisfies the assignment with a Sri Lankan public dataset, traditional ML comparison, SHAP explainability, and a deployable Streamlit front-end.
