"""
FlowCast — Financial analytics (native Streamlit UI).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as ch
from data_processor import load_and_process
from forecaster import ForecastMethod, get_forecast, stl_decomposition_monthly

st.set_page_config(
    page_title="FlowCast",
    page_icon="📈",
    layout="wide",
)

NO_DATA_MSG = (
    "Upload a bank export (CSV or Excel) in the sidebar to get started. "
    "Your file should include date and amount columns; a category column unlocks richer charts."
)


# ---------------------------------------------------------------------------
# Session state & helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "mode": "personal",
        "df_raw": None,
        "df_monthly": None,
        "summary": None,
        "file_name": None,
        "file_bytes": None,
        "fc_prophet": None,
        "fc_arima": None,
        "fc_fingerprint": None,
        "fc_horizon": 6,
        "fc_ci": 0.95,
        "fc_season": "auto",
        "fc_whatif": 0.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _has_data() -> bool:
    return st.session_state.df_monthly is not None and st.session_state.df_raw is not None


def fmt_money(x: float | None, *, compact: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    x = float(x)
    ax = abs(x)
    if compact and ax >= 1_000_000:
        return f"${x / 1_000_000:,.1f}M"
    if compact and ax >= 1_000:
        return f"${x / 1_000:,.1f}K"
    return f"${x:,.2f}"


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞%"
    return f"{x:+.1f}%"


def _safe_plot(fn: Callable[..., go.Figure], *args: Any, **kwargs: Any) -> go.Figure | None:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _plot_chart(fn: Callable[..., go.Figure], *args: Any, **kwargs: Any) -> None:
    fig = _safe_plot(fn, *args, **kwargs)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Chart could not be drawn for this data.")


def _load_from_upload(uploaded: Any) -> None:
    st.session_state.file_bytes = uploaded.getvalue()
    st.session_state.file_name = uploaded.name
    fn = st.session_state.file_name or "data.csv"
    with st.spinner("Analyzing your data…"):
        df_m, df_r, summary, _mode = load_and_process(
            io.BytesIO(st.session_state.file_bytes),
            mode=st.session_state.mode,
            filename=fn,
        )
    st.session_state.df_monthly = df_m
    st.session_state.df_raw = df_r
    st.session_state.summary = summary
    st.session_state.fc_prophet = None
    st.session_state.fc_arima = None
    st.session_state.fc_fingerprint = None


def _reprocess_if_mode_changed() -> None:
    b = st.session_state.file_bytes
    if not b:
        return
    fn = st.session_state.file_name or "data.csv"
    with st.spinner("Updating for mode…"):
        df_m, df_r, summary, _ = load_and_process(
            io.BytesIO(b),
            mode=st.session_state.mode,
            filename=fn,
        )
    st.session_state.df_monthly = df_m
    st.session_state.df_raw = df_r
    st.session_state.summary = summary
    st.session_state.fc_prophet = None
    st.session_state.fc_arima = None
    st.session_state.fc_fingerprint = None


def _forecast_fingerprint() -> str:
    return "|".join(
        str(st.session_state.get(k, ""))
        for k in ("mode", "fc_horizon", "fc_ci", "fc_season", "fc_whatif")
    )


def _holdout_pred(monthly: pd.DataFrame, method: ForecastMethod, **kw: Any) -> np.ndarray | None:
    if len(monthly) < 6:
        return None
    train = monthly.iloc[:-3].reset_index(drop=True)
    act = monthly.iloc[-3:]["y"].to_numpy(dtype=float)
    try:
        fc, _ = get_forecast(train, method=method, periods=3, **kw)
        return fc["yhat"].to_numpy(dtype=float)[:3]
    except Exception:
        return None


def _accuracy_block(
    monthly: pd.DataFrame, method: ForecastMethod, **kw: Any
) -> tuple[float | None, float | None, float | None, float | None]:
    if len(monthly) < 6:
        return (None,) * 4
    y = monthly.iloc[-3:]["y"].to_numpy(dtype=float)
    pred = _holdout_pred(monthly, method, **kw)
    if pred is None:
        return (None,) * 4
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mask = np.abs(y) > 1e-9
    mape = float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100.0) if mask.any() else None
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, mape, r2


def _health_score(y: np.ndarray) -> float:
    if len(y) < 2:
        return 50.0
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    t_pts = 25 * (1.0 if slope <= 0 else max(0.0, 1.0 - min(slope / (np.mean(y) + 1e-9), 1.0)))
    cv = float(np.std(y) / (np.mean(y) + 1e-9))
    c_pts = max(0.0, 25 - min(cv * 80, 25))
    mu = float(np.mean(y))
    sd = float(np.std(y)) or 1.0
    ano = int(np.sum(y > mu + 1.5 * sd))
    a_pts = max(0.0, 25 - min(ano * 8, 25))
    return float(np.clip(t_pts + c_pts + a_pts + 25.0, 0, 100))


def _require_data() -> bool:
    if _has_data():
        return True
    st.info(NO_DATA_MSG)
    return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> None:
    st.title("FlowCast")
    st.caption("Financial Intelligence Platform")

    mode_label = st.radio(
        "Mode",
        options=["Personal", "Business"],
        index=0 if st.session_state.mode == "personal" else 1,
        horizontal=True,
    )
    new_mode = "personal" if mode_label == "Personal" else "business"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        _reprocess_if_mode_changed()

    st.subheader("Data")
    uploaded = st.file_uploader(
        "Upload bank export",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel with date and amount columns.",
    )
    if uploaded is not None:
        _load_from_upload(uploaded)

    if st.session_state.file_name:
        st.success(f"✅ {st.session_state.file_name}")
    else:
        sample = Path(__file__).resolve().parent / "sample_data.csv"
        if sample.is_file():
            st.download_button(
                "Download sample CSV",
                sample.read_bytes(),
                file_name="sample_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if st.session_state.file_name and st.button("Clear data", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def tab_dashboard() -> None:
    if not _require_data():
        return

    st.title("FlowCast 📈")
    s = st.session_state.summary or {}
    dm = st.session_state.df_monthly
    raw = st.session_state.df_raw
    personal = st.session_state.mode == "personal"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total in window", fmt_money(float(s.get("total") or 0)))
    with c2:
        st.metric("Average per month", fmt_money(float(s.get("mean") or 0)))
    with c3:
        st.metric("Peak month", str(s.get("max_month") or "—"), fmt_money(float(s.get("max_val") or 0)))
    with c4:
        st.metric("3‑month change", fmt_pct(float(s.get("pct_change")) if s.get("pct_change") is not None else None))

    left, right = st.columns(2)
    with left:
        trend_title = "Monthly spending trend" if personal else "Monthly net trend"
        _plot_chart(ch.plot_area_trend, dm, title=trend_title)
    with right:
        try:
            cat_ok = raw["category"].astype(str).str.strip().ne("") & (
                raw["category"].astype(str).str.lower().ne("nan")
            )
            if cat_ok.any():
                gcat = raw.loc[cat_ok].copy()
                gcat["_a"] = gcat["amount"].abs()
                g = gcat.groupby("category", as_index=False)["_a"].sum()
                g = g.sort_values("_a", ascending=False).head(8)
                _plot_chart(
                    ch.plot_donut,
                    g["category"].tolist(),
                    g["_a"].tolist(),
                    title="Share by category",
                )
            else:
                total = float(raw["amount"].abs().sum())
                _plot_chart(ch.plot_donut, ["Total"], [total], title="Total spend")
        except Exception as e:
            st.warning(f"Category chart skipped: {e}")

    st.subheader("Monthly breakdown")
    bar_title = "Monthly spending" if personal else "Monthly net"
    _plot_chart(ch.plot_monthly_bars, dm, title=bar_title)

    with st.expander("Summary details"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Median month",
                        "Transactions",
                        "Avg per transaction",
                        "Most common category",
                        "Trend",
                    ],
                    "Value": [
                        fmt_money(float(s.get("median") or 0)),
                        str(int(s.get("total_transactions") or 0)),
                        fmt_money(float(s.get("avg_per_transaction") or 0)),
                        str(s.get("most_common_category") or "—"),
                        str(s.get("trend", "—")).title(),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def tab_spending() -> None:
    if not _require_data():
        return

    st.header("💸 Spending")
    raw = st.session_state.df_raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])

    dmin, dmax = raw["date"].min().date(), raw["date"].max().date()
    dr = st.date_input("Date range", value=(dmin, dmax))
    if isinstance(dr, tuple) and len(dr) == 2:
        m1, m2 = dr
        tx = raw.loc[(raw["date"].dt.date >= m1) & (raw["date"].dt.date <= m2)].copy()
    else:
        tx = raw.copy()

    c1, c2 = st.columns(2)
    with c1:
        if tx["category"].astype(str).str.strip().ne("").any():
            _plot_chart(ch.plot_treemap, tx, "category", "amount", title="Spending by category")
        else:
            st.info("Add a category column in your file to see the category treemap.")
    with c2:
        _plot_chart(ch.plot_day_of_week, tx, "date", "amount", title="Average spend by weekday")

    st.subheader("Top merchants / labels")
    tx["_merchant"] = tx["category"].where(
        tx["category"].astype(str).str.strip().ne(""), "Unknown"
    )
    tx["_amt"] = tx["amount"].abs()
    merchants = (
        tx.groupby("_merchant", as_index=False)["_amt"]
        .sum()
        .rename(columns={"_merchant": "merchant", "_amt": "total"})
        .sort_values("total", ascending=False)
        .head(15)
    )
    merchants["total"] = merchants["total"].map(lambda v: fmt_money(float(v)))
    st.dataframe(merchants, use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        _plot_chart(ch.plot_heatmap_calendar, tx, "date", "amount", title="Calendar heatmap")
    with c4:
        _plot_chart(ch.plot_histogram, tx, "amount", title="Transaction size distribution")

    st.subheader("Largest transactions")
    top = (
        tx.assign(abs_amt=tx["amount"].abs())
        .sort_values("abs_amt", ascending=False)
        .head(10)[["date", "category", "amount"]]
        .copy()
    )
    top["date"] = top["date"].dt.strftime("%Y-%m-%d")
    top["amount"] = top["amount"].map(lambda v: fmt_money(float(v)))
    st.dataframe(top, use_container_width=True, hide_index=True)


def tab_forecast() -> None:
    if not _require_data():
        return

    st.header("🔮 Forecast")
    dm = st.session_state.df_monthly.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox("Model", ["Prophet", "ARIMA", "Both"], index=2)
    with c2:
        periods = st.slider("Months ahead", 3, 24, int(st.session_state.fc_horizon))
        st.session_state.fc_horizon = periods
    with c3:
        ci_label = st.selectbox("Confidence band", ["80%", "90%", "95%"], index=2)
        st.session_state.fc_ci = {"80%": 0.80, "90%": 0.90, "95%": 0.95}[ci_label]

    seas = st.selectbox("Seasonality", ["Auto", "Weekly", "Monthly", "Yearly"], index=0)
    st.session_state.fc_season = {
        "Auto": "auto",
        "Weekly": "weekly",
        "Monthly": "monthly",
        "Yearly": "yearly",
    }[seas]
    st.session_state.fc_whatif = st.slider("What-if adjustment on history (%)", -30, 30, int(st.session_state.fc_whatif))

    kw: dict[str, Any] = dict(
        interval_width=float(st.session_state.fc_ci),
        seasonality=st.session_state.fc_season,
    )
    dm_adj = dm.copy()
    dm_adj["y"] = dm_adj["y"].astype(float) * (1.0 + st.session_state.fc_whatif / 100.0)

    cur_fp = _forecast_fingerprint()
    run = st.button("Run forecast", type="primary")
    if run or st.session_state.get("fc_fingerprint") != cur_fp or st.session_state.fc_prophet is None:
        with st.spinner("Running forecast…"):
            try:
                p_df, _ = get_forecast(dm_adj, "prophet", periods, **kw)
                a_df, _ = get_forecast(dm_adj, "arima", periods, **kw)
                st.session_state.fc_prophet = p_df
                st.session_state.fc_arima = a_df
                st.session_state.fc_fingerprint = cur_fp
            except Exception as e:
                st.error(str(e))

    p_df = st.session_state.fc_prophet
    a_df = st.session_state.fc_arima
    if p_df is None or a_df is None:
        st.info("Tap **Run forecast** to generate projections.")
        return

    if model == "Both":
        _plot_chart(
            ch.plot_forecast,
            dm_adj,
            p_df,
            title="Forecast (Prophet + ARIMA)",
            forecast_arima=a_df,
        )
    elif model == "Prophet":
        _plot_chart(ch.plot_forecast, dm_adj, p_df, title="Prophet forecast", forecast_arima=None)
    else:
        _plot_chart(ch.plot_forecast, dm_adj, a_df, title="ARIMA forecast", forecast_arima=None)

    method: ForecastMethod = "prophet" if model != "ARIMA" else "arima"
    mae, rmse, mape, r2 = _accuracy_block(dm_adj, method, **kw)

    st.subheader("Accuracy (last 3 months holdout)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MAE", fmt_money(mae) if mae is not None else "—")
    with m2:
        st.metric("RMSE", fmt_money(rmse) if rmse is not None else "—")
    with m3:
        st.metric("MAPE", f"{mape:.1f}%" if mape is not None else "—")
    with m4:
        st.metric("R²", f"{r2:.3f}" if r2 is not None else "—")

    with st.expander("Forecast table"):
        show = p_df.copy()
        if model == "Both":
            show = show.merge(a_df.rename(columns={"yhat": "yhat_arima"}), on="ds", how="outer")
        st.dataframe(show, use_container_width=True)

    with st.expander("Seasonal decomposition (STL)"):
        try:
            dec = stl_decomposition_monthly(dm_adj)
            _plot_chart(ch.plot_decomposition, dec)
        except Exception as e:
            st.info(f"Seasonal view unavailable: {e}")


def tab_insights() -> None:
    if not _require_data():
        return

    st.header("💡 Insights")
    s = st.session_state.summary or {}
    dm = st.session_state.df_monthly
    raw = st.session_state.df_raw
    y = dm["y"].to_numpy(dtype=float)
    score = _health_score(y)

    st.subheader("Financial health score")
    _plot_chart(ch.plot_gauge, score, "Financial health score", max_val=100.0)

    st.subheader("Key takeaways")
    bullets = [
        f"Trend is **{str(s.get('trend', '—')).title()}** with **{fmt_pct(float(s.get('pct_change')))}** "
        "comparing first vs last three-month averages.",
        f"Largest month: **{s.get('max_month', '—')}** at **{fmt_money(float(s.get('max_val') or 0))}**.",
        f"Most common category: **{s.get('most_common_category') or 'n/a'}**.",
        f"**{int(s.get('total_transactions') or 0)}** transactions; "
        f"about **{fmt_money(float(s.get('avg_per_transaction') or 0))}** per transaction on average.",
        f"Typical month-to-month swing: **{fmt_money(float(s.get('std') or 0))}**.",
    ]
    for b in bullets:
        st.write(f"- {b}")

    st.subheader("Recommendations")
    st.warning("Pay attention: Review your highest month and set a simple monthly spending ceiling.")
    st.warning("Watch: Each Friday, compare week-to-date spend vs last week.")
    st.success("Low effort: Audit one subscription—small recurring charges add up.")

    num = raw.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        st.subheader("Numeric column relationships")
        _plot_chart(ch.plot_correlation, num, title="Correlation matrix")

    days = (pd.to_datetime(raw["date"]).max() - pd.to_datetime(raw["date"]).min()).days + 1
    per_day = float(raw["amount"].abs().sum() / max(days, 1))
    raw2 = raw.copy()
    raw2["_dow"] = pd.to_datetime(raw2["date"]).dt.day_name()
    topd = raw2.groupby("_dow")["amount"].apply(lambda x: x.abs().sum()).idxmax()
    st.subheader("Did you know?")
    st.write(f"- Average per day in your window: **{fmt_money(per_day)}**")
    st.write(f"- Busiest weekday by total dollars: **{topd}**")


def tab_export() -> None:
    if not _require_data():
        return

    st.header("📥 Export")

    dm = st.session_state.df_monthly
    raw = st.session_state.df_raw

    st.subheader("Monthly summary")
    st.download_button(
        "Download monthly CSV",
        dm.to_csv(index=False).encode("utf-8"),
        file_name="flowcast_monthly.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Transactions")
    st.download_button(
        "Download transactions CSV",
        raw.to_csv(index=False).encode("utf-8"),
        file_name="flowcast_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Forecast")
    p_df = st.session_state.fc_prophet
    if p_df is not None:
        st.download_button(
            "Download forecast CSV",
            p_df.to_csv(index=False).encode("utf-8"),
            file_name="flowcast_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Run a forecast on the Forecast tab to enable forecast download.")

    st.subheader("Insights summary")
    lines = [f"{k}: {v}" for k, v in (st.session_state.summary or {}).items()]
    st.download_button(
        "Download insights TXT",
        "\n".join(lines).encode("utf-8"),
        file_name="flowcast_insights.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.subheader("Excel workbook")
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            dm.to_excel(w, sheet_name="Monthly", index=False)
            raw.to_excel(w, sheet_name="Transactions", index=False)
            if p_df is not None:
                p_df.to_excel(w, sheet_name="Forecast", index=False)
        st.download_button(
            "Download Excel workbook",
            buf.getvalue(),
            file_name="flowcast_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Excel export needs openpyxl: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()
    _sidebar()

    tab_dash, tab_spend, tab_fc, tab_ins, tab_exp = st.tabs(
        [
            "📊 Dashboard",
            "💸 Spending",
            "🔮 Forecast",
            "💡 Insights",
            "📥 Export",
        ]
    )

    with tab_dash:
        tab_dashboard()
    with tab_spend:
        tab_spending()
    with tab_fc:
        tab_forecast()
    with tab_ins:
        tab_insights()
    with tab_exp:
        tab_export()


if __name__ == "__main__":
    main()
