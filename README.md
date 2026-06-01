# FlowCast : Python Analytics Backend

> Time series forecasting engine powering the FlowCast personal finance platform.

---

## What It Does

This is the Python backend for FlowCast. It handles all forecasting and statistical analysis:

- **Prophet forecasting** — trend decomposition with seasonal patterns
- **ARIMA forecasting** — seasonal ARIMA with widening confidence intervals
- **Auto model selection** — holdout RMSE comparison picks the best model automatically
- **STL decomposition** — trend, seasonal, and residual breakdown on monthly spend data
- **Data validation** — cleans and normalizes transaction DataFrames before modeling

---

## File Structure

```
FlowCast/
├── app.py                 # Streamlit app entry point
├── forecaster.py          # Prophet, ARIMA, auto-selection, STL decomposition
├── data_processor.py      # CSV parsing, category tagging, aggregation
├── charts.py              # Visualization helpers
├── requirements.txt       # Python dependencies
└── sample_data.csv        # 256 Chase transactions (Jan–Dec 2024) for testing
```

---

## Forecasting Models

| Model | Method | Best For |
|---|---|---|
| Prophet | Trend decomposition + seasonality | Data with strong seasonal patterns |
| ARIMA (2,1,2) | Seasonal ARIMA | Smooth baseline with gradual trend |
| Auto | Holdout RMSE comparison | Automatically picks best model |

Auto mode splits data into train/holdout (last 3 months), scores both models on RMSE, and selects the winner. Falls back to Prophet if ARIMA fails.

---

## Core Function

```python
from forecaster import get_forecast

# df must have 'ds' (date) and 'y' (spend amount) columns
forecast_df, model_used = get_forecast(
    df,
    method="auto",      # "prophet", "arima", or "auto"
    periods=6,          # months ahead
    interval_width=0.95
)
```

Returns a DataFrame with `ds`, `yhat`, `yhat_lower`, `yhat_upper` columns.

---

## STL Decomposition

```python
from forecaster import stl_decomposition_monthly

decomp = stl_decomposition_monthly(df)
# Returns: ds, trend, seasonal, resid
```

---

## Getting Started

**Requirements: Python 3.9+**

```bash
# 1. Clone the repo
git clone https://github.com/samettemurcin/FlowCast.git
cd FlowCast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## Sample Data

`sample_data.csv` contains 256 real-format Chase transactions from Jan–Dec 2024 for local testing.

---

## Roadmap

- [ ] Connect live inference to React frontend (API endpoint)
- [ ] Add ETS (Exponential Smoothing) model
- [ ] Multi-category forecasting (per-category spend projection)
- [ ] Anomaly detection model (Isolation Forest)
- [ ] Docker container for deployment

---

## Related

[tryflowcast](https://github.com/samettemurcin/tryflowcast) — React frontend (TypeScript, Tailwind, Cloudflare Workers)

---

## Author

**Samet Temurcin**
MS Data Analytics Engineering — Northeastern University
[linkedin.com/in/samet-temurcin](https://linkedin.com/in/samet-temurcin) · [github.com/samettemurcin](https://github.com/samettemurcin)
