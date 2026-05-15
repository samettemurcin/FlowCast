"""
FlowCast — forecasting with Prophet and statsmodels ARIMA.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

ForecastMethod = Literal["prophet", "arima", "auto"]
SeasonalityPreset = Literal["auto", "weekly", "monthly", "yearly"]


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")
    for col in ("ds", "y"):
        if col not in df.columns:
            raise ValueError(f"df must contain column '{col}'.")
    out = df[["ds", "y"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"]).sort_values("ds", ignore_index=True)
    if out.empty:
        raise ValueError("No valid ds/y rows after cleaning.")
    return out


def _prophet_seasonality(preset: SeasonalityPreset) -> dict:
    if preset == "weekly":
        return dict(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    if preset == "monthly":
        return dict(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    if preset == "yearly":
        return dict(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    return dict(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)


def _forecast_prophet(
    df: pd.DataFrame,
    periods: int,
    *,
    interval_width: float = 0.95,
    seasonality: SeasonalityPreset = "auto",
) -> pd.DataFrame:
    sk = _prophet_seasonality(seasonality)
    m = Prophet(interval_width=interval_width, **sk)
    m.fit(df)
    try:
        future = m.make_future_dataframe(
            periods=periods, freq="MS", include_history=False
        )
    except TypeError:
        future = m.make_future_dataframe(periods=periods, freq="MS")
    fcst_all = m.predict(future)
    cutoff = pd.Timestamp(df["ds"].max())
    fcst = fcst_all[fcst_all["ds"] > cutoff].head(periods)
    fcst = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    if len(fcst) < periods:
        raise RuntimeError(
            f"Prophet returned only {len(fcst)} future rows; expected {periods}."
        )
    fcst["ds"] = pd.to_datetime(fcst["ds"])
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        fcst[c] = fcst[c].astype(float)
    return fcst.reset_index(drop=True)


def _future_month_starts(last_ds: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    last_period = pd.Timestamp(last_ds).to_period("M")
    first_future = (last_period + 1).to_timestamp()
    return pd.date_range(first_future, periods=periods, freq="MS")


def _forecast_arima(
    df: pd.DataFrame,
    periods: int,
    *,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    if periods < 1:
        raise ValueError("periods must be at least 1.")
    alpha = max(0.001, min(0.5, 1.0 - interval_width))
    model = ARIMA(df["y"].astype(float), order=(2, 1, 2))
    fitted = model.fit()
    fr = fitted.get_forecast(steps=periods)
    yhat = fr.predicted_mean.to_numpy(dtype=float)
    ci = fr.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0].to_numpy(dtype=float)
    upper = ci.iloc[:, 1].to_numpy(dtype=float)
    ds = _future_month_starts(df["ds"].max(), periods)
    if len(ds) != len(yhat):
        raise RuntimeError("ARIMA forecast length does not match future date index.")
    out = pd.DataFrame(
        {
            "ds": ds,
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
        }
    )
    out["ds"] = pd.to_datetime(out["ds"]).astype("datetime64[ns]")
    return out


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(predicted, dtype=float)
    if a.shape != b.shape:
        raise ValueError("actual and predicted must have the same shape.")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _holdout_rmse_prophet(
    train: pd.DataFrame,
    actual_y: np.ndarray,
    *,
    interval_width: float = 0.95,
    seasonality: SeasonalityPreset = "auto",
) -> float:
    fcst = _forecast_prophet(
        train, periods=len(actual_y), interval_width=interval_width, seasonality=seasonality
    )
    pred = fcst["yhat"].to_numpy(dtype=float)[: len(actual_y)]
    return _rmse(actual_y, pred)


def _holdout_rmse_arima(
    train: pd.DataFrame,
    actual_y: np.ndarray,
    *,
    interval_width: float = 0.95,
) -> float:
    fcst = _forecast_arima(train, periods=len(actual_y), interval_width=interval_width)
    pred = fcst["yhat"].to_numpy(dtype=float)[: len(actual_y)]
    return _rmse(actual_y, pred)


def _pick_auto_method(
    df: pd.DataFrame,
    *,
    interval_width: float = 0.95,
    seasonality: SeasonalityPreset = "auto",
) -> Literal["prophet", "arima"]:
    holdout_months = 3
    min_total = holdout_months + 4
    if len(df) < min_total:
        return "prophet"

    train = df.iloc[:-holdout_months].reset_index(drop=True)
    actual_y = df.iloc[-holdout_months:]["y"].to_numpy(dtype=float)

    rmse_p = np.inf
    rmse_a = np.inf
    try:
        rmse_p = _holdout_rmse_prophet(
            train, actual_y, interval_width=interval_width, seasonality=seasonality
        )
    except Exception:
        rmse_p = np.inf
    try:
        rmse_a = _holdout_rmse_arima(train, actual_y, interval_width=interval_width)
    except Exception:
        rmse_a = np.inf

    if not np.isfinite(rmse_p) and not np.isfinite(rmse_a):
        raise ValueError(
            "Auto mode could not score Prophet or ARIMA on holdout data; "
            "try more history or use method='prophet' / 'arima' explicitly."
        )
    if rmse_p <= rmse_a:
        return "prophet"
    return "arima"


def get_forecast(
    df: pd.DataFrame,
    method: ForecastMethod = "prophet",
    periods: int = 6,
    *,
    interval_width: float = 0.95,
    seasonality: SeasonalityPreset = "auto",
) -> Tuple[pd.DataFrame, str]:
    """
    Multi-step monthly forecast. ``interval_width`` applies to Prophet and ARIMA bands.
    ``seasonality`` presets adjust Prophet seasonality flags (monthly data).
    """
    if periods < 1:
        raise ValueError("periods must be at least 1.")
    if not 0.5 < interval_width < 1.0:
        raise ValueError("interval_width should be between 0.5 and 1.0.")

    data = _validate_df(df)

    if method == "auto":
        chosen = _pick_auto_method(data, interval_width=interval_width, seasonality=seasonality)
        try:
            if chosen == "prophet":
                return (
                    _forecast_prophet(
                        data, periods, interval_width=interval_width, seasonality=seasonality
                    ),
                    "prophet",
                )
            return _forecast_arima(data, periods, interval_width=interval_width), "arima"
        except Exception:
            if chosen == "prophet":
                return _forecast_arima(data, periods, interval_width=interval_width), "arima"
            return (
                _forecast_prophet(
                    data, periods, interval_width=interval_width, seasonality=seasonality
                ),
                "prophet",
            )

    if method == "prophet":
        try:
            return (
                _forecast_prophet(
                    data, periods, interval_width=interval_width, seasonality=seasonality
                ),
                "prophet",
            )
        except Exception:
            return _forecast_arima(data, periods, interval_width=interval_width), "arima"

    if method == "arima":
        return _forecast_arima(data, periods, interval_width=interval_width), "arima"

    raise ValueError("method must be 'prophet', 'arima', or 'auto'.")


def stl_decomposition_monthly(df: pd.DataFrame, seasonal: int | None = None) -> pd.DataFrame:
    """STL decomposition on monthly ``y``; returns columns ds, trend, seasonal, resid."""
    d = _validate_df(df).copy()
    d = d.set_index("ds").asfreq("MS")
    y = d["y"].astype(float)
    y = y.interpolate(limit_direction="both")
    period = seasonal or min(12, max(3, len(y) // 2 * 2))
    if len(y) < period * 2:
        period = max(3, len(y) // 2 | 1)
    stl = STL(y, period=period, robust=True)
    res = stl.fit()
    out = pd.DataFrame(
        {
            "ds": y.index,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        }
    ).reset_index(drop=True)
    return out
