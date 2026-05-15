"""
FlowCast — Plotly charts (Plotly 6.x safe: no fig.layout dict iteration).
Build with go.Figure / traces; style with update_layout, update_xaxes, update_yaxes, update_traces.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "Inter, sans-serif"
C_BG = "#FFFFFF"
C_GRID = "#F0F0F0"
C_TEXT = "#1A1A2E"
C_PRIMARY = "#6C63FF"
C_SECONDARY = "#00C9A7"
C_ACCENT = "#FF6584"
C_WARNING = "#FFB347"
C_SUCCESS = "#00B894"
C_DANGER = "#E17055"
C_BLUE = "#0984E3"
C_PURPLE_LIGHT = "#A29BFE"

_QUALITATIVE = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _trace_type(fig: go.Figure) -> str | None:
    if not fig.data:
        return None
    return getattr(fig.data[0], "type", None)


def _finalize(fig: go.Figure, title: str | None = None, *, axes: bool = True) -> go.Figure:
    """Apply light theme via update_layout / update_axes only."""
    t = _trace_type(fig)
    layout: dict[str, Any] = dict(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(family=FONT, color=C_TEXT, size=13),
        margin=dict(l=52, r=40, t=56 if title else 44, b=48),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", borderwidth=0, font=dict(color=C_TEXT)),
        hoverlabel=dict(font_family=FONT, font_size=13),
        showlegend=True,
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=17, color=C_TEXT), x=0.02, xanchor="left")
    fig.update_layout(**layout)

    if axes and t not in ("pie", "treemap", "indicator", "table"):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=C_GRID,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#E2E8F0",
            tickfont=dict(color=C_TEXT),
            title_font=dict(color=C_TEXT),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=C_GRID,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#E2E8F0",
            tickfont=dict(color=C_TEXT),
            title_font=dict(color=C_TEXT),
        )
    return fig


def plot_area_trend(df: pd.DataFrame, title: str = "Monthly trend") -> go.Figure:
    d = df.sort_values("ds").copy()
    d["ds"] = pd.to_datetime(d["ds"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["ds"],
            y=d["y"],
            mode="lines",
            line=dict(color=C_PRIMARY, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.18)",
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(title_text="Month")
    return _finalize(fig, title)


def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = "Forecast",
    *,
    forecast_arima: pd.DataFrame | None = None,
    hist_name: str = "Historical",
) -> go.Figure:
    hist = historical_df.sort_values("ds").copy()
    hist["ds"] = pd.to_datetime(hist["ds"])
    fc = forecast_df.sort_values("ds").copy()
    fc["ds"] = pd.to_datetime(fc["ds"])
    boundary = pd.Timestamp(hist["ds"].max())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(108,99,255,0.2)",
            fill="tonexty",
            name="Forecast range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            name=hist_name,
            line=dict(color=C_BLUE, width=2.8),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color=C_PRIMARY, width=2.5),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    if forecast_arima is not None and not forecast_arima.empty:
        fa = forecast_arima.sort_values("ds").copy()
        fa["ds"] = pd.to_datetime(fa["ds"])
        fig.add_trace(
            go.Scatter(
                x=fa["ds"],
                y=fa["yhat"],
                mode="lines",
                name="ARIMA",
                line=dict(color=C_WARNING, width=2.5, dash="dash"),
                hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    fig.add_vline(x=boundary, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.update_yaxes(tickprefix="$")
    fig.update_layout(hovermode="x unified")
    return _finalize(fig, title)


def plot_donut(labels: Sequence[str], values: Sequence[float], title: str = "Distribution") -> go.Figure:
    n = len(labels)
    colors = [_QUALITATIVE[i % len(_QUALITATIVE)] for i in range(n)]
    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=C_BG, width=2)),
            textinfo="percent+label",
            hovertemplate="%{label}<br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(title=title)
    return _finalize(fig, axes=False)


def plot_treemap(df: pd.DataFrame, cat_col: str, val_col: str, title: str = "Treemap") -> go.Figure:
    d = df[[cat_col, val_col]].dropna().copy()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna()
    agg = d.groupby(cat_col, as_index=False)[val_col].sum()
    labels = agg[cat_col].astype(str).tolist()
    parents = [""] * len(labels)
    values = agg[val_col].astype(float).tolist()
    vmax = max(values) if values else 1.0
    norm = [float(v) / vmax for v in values]
    colors = [f"rgba(108,99,255,{0.25 + 0.65 * t})" for t in norm]
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, line=dict(width=1, color="#fff")),
            textinfo="label+value+percent parent",
            hovertemplate="%{label}<br>$%{value:,.2f}<extra></extra>",
        )
    )
    return _finalize(fig, title, axes=False)


def plot_heatmap_calendar(df: pd.DataFrame, date_col: str, amount_col: str, title: str = "Spend heatmap") -> go.Figure:
    t = df[[date_col, amount_col]].dropna().copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t[amount_col] = pd.to_numeric(t[amount_col], errors="coerce").abs()
    t = t.dropna()
    if t.empty:
        raise ValueError("No data for heatmap.")
    t["_dom"] = t[date_col].dt.day
    t["_ym"] = t[date_col].dt.strftime("%Y-%m")
    g = t.groupby(["_ym", "_dom"], as_index=False)[amount_col].sum()
    piv = g.pivot(index="_ym", columns="_dom", values=amount_col).fillna(0.0)
    xs = [str(c) for c in piv.columns.tolist()]
    ys = [str(i) for i in piv.index.tolist()]
    z = piv.values.tolist()
    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=z,
            colorscale=[[0, "#f8f9ff"], [0.5, C_PURPLE_LIGHT], [1, C_PRIMARY]],
            hovertemplate="Day %{x}<br>%{y}<br>$%{z:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Day of month", yaxis_title="Month")
    return _finalize(fig, title)


def plot_horizontal_bars(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str = "Top merchants",
    n: int = 10,
) -> go.Figure:
    d = df[[label_col, value_col]].dropna().copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").abs()
    d = d.groupby(label_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(n)
    fig = go.Figure(
        go.Bar(
            x=d[value_col],
            y=d[label_col],
            orientation="h",
            marker=dict(color=C_PRIMARY),
            text=[f"${v:,.0f}" for v in d[value_col]],
            textposition="outside",
            hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Total ($)")
    return _finalize(fig, title)


def plot_gauge(value: float, title: str = "Score", max_val: float = 100.0) -> go.Figure:
    v = float(np.clip(value, 0, max_val))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            title={"text": title, "font": dict(size=14, color=C_TEXT)},
            number=dict(font=dict(size=28, color=C_TEXT)),
            gauge={
                "axis": {"range": [0, max_val]},
                "bar": {"color": C_PRIMARY},
                "steps": [
                    {"range": [0, 40], "color": "#E8FFF5"},
                    {"range": [40, 70], "color": "#FFF8E8"},
                    {"range": [70, 100], "color": "#FFF0F0"},
                ],
            },
        )
    )
    return _finalize(fig, axes=False)


def plot_histogram(df: pd.DataFrame, amount_col: str, title: str = "Amount distribution") -> go.Figure:
    s = pd.to_numeric(df[amount_col], errors="coerce").dropna().abs()
    fig = go.Figure(
        go.Histogram(
            x=s,
            nbinsx=40,
            marker=dict(color=C_PRIMARY, line=dict(color=C_BG, width=1)),
            hovertemplate="$%{x:,.2f}<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Count", showlegend=False)
    return _finalize(fig, title)


def plot_waterfall(labels: list[str], values: list[float], title: str = "Waterfall") -> go.Figure:
    n = len(values)
    if n == 0:
        raise ValueError("Waterfall needs values.")
    if n == 1:
        measure = ["total"]
    elif n == 2:
        measure = ["absolute", "total"]
    else:
        measure = ["absolute"] + ["relative"] * (n - 2) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            name="cf",
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            connector={"line": {"color": "#cbd5e1"}},
            increasing={"marker": {"color": C_SECONDARY}},
            decreasing={"marker": {"color": C_DANGER}},
            totals={"marker": {"color": C_BLUE}},
        )
    )
    fig.update_yaxes(tickprefix="$")
    return _finalize(fig, title)


def plot_decomposition(decomp: pd.DataFrame, title: str = "STL decomposition") -> go.Figure:
    d = decomp.copy()
    d["ds"] = pd.to_datetime(d["ds"])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Trend", "Seasonal", "Residual"),
    )
    fig.add_trace(go.Scatter(x=d["ds"], y=d["trend"], mode="lines", line=dict(color=C_BLUE), name="Trend"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=d["ds"], y=d["seasonal"], mode="lines", line=dict(color=C_SECONDARY), name="Seasonal"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d["ds"], y=d["resid"], mode="lines", line=dict(color=C_ACCENT), name="Residual"),
        row=3,
        col=1,
    )
    fig.update_layout(height=720, title_text=title, showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=False)
    return _finalize(fig)


def plot_scatter(df: pd.DataFrame, date_col: str, amount_col: str, color_col: str | None, title: str) -> go.Figure:
    t = df.copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t[amount_col] = pd.to_numeric(t[amount_col], errors="coerce")
    t = t.dropna(subset=[date_col, amount_col])
    fig = go.Figure()
    if color_col and color_col in t.columns:
        cats = t[color_col].astype(str).unique().tolist()
        amax = float(t[amount_col].abs().max()) or 1.0
        for i, cat in enumerate(cats):
            sub = t.loc[t[color_col].astype(str) == cat]
            sizes = np.clip(sub[amount_col].abs().to_numpy(dtype=float) / amax * 28.0 + 6.0, 6.0, 36.0)
            fig.add_trace(
                go.Scatter(
                    x=sub[date_col],
                    y=sub[amount_col],
                    mode="markers",
                    name=str(cat)[:40],
                    marker=dict(
                        size=sizes,
                        color=_QUALITATIVE[i % len(_QUALITATIVE)],
                        opacity=0.85,
                        line=dict(width=0.5, color="#fff"),
                    ),
                    hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
                )
            )
    else:
        amax = float(t[amount_col].abs().max()) or 1.0
        sizes = np.clip(t[amount_col].abs().to_numpy(dtype=float) / amax * 28.0 + 6.0, 6.0, 36.0)
        fig.add_trace(
            go.Scatter(
                x=t[date_col],
                y=t[amount_col],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=C_PRIMARY,
                    opacity=0.85,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(xaxis_title="Date", yaxis_title="Amount ($)")
    return _finalize(fig, title)


def plot_correlation(df: pd.DataFrame, title: str = "Correlation matrix") -> go.Figure:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns.")
    c = num.corr(numeric_only=True)
    cols = c.columns.tolist()
    fig = go.Figure(
        go.Heatmap(
            z=c.values.tolist(),
            x=cols,
            y=cols,
            zmin=-1,
            zmax=1,
            colorscale="Blues",
            text=np.round(c.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}<br>r=%{z:.3f}<extra></extra>",
        )
    )
    return _finalize(fig, title)


def plot_day_of_week(df: pd.DataFrame, date_col: str, amount_col: str, title: str = "Spend by weekday") -> go.Figure:
    t = df[[date_col, amount_col]].dropna().copy()
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
    t[amount_col] = pd.to_numeric(t[amount_col], errors="coerce").abs()
    t = t.dropna()
    t["_dow"] = t[date_col].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = t.groupby("_dow", as_index=False)[amount_col].mean()
    g["_dow"] = pd.Categorical(g["_dow"], categories=order, ordered=True)
    g = g.sort_values("_dow")
    fig = go.Figure(
        go.Bar(
            x=[str(x) for x in g["_dow"].tolist()],
            y=g[amount_col].astype(float),
            marker=dict(color=C_PRIMARY),
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Weekday", yaxis_title="Average spend ($)")
    return _finalize(fig, title)


def plot_boxplot(df: pd.DataFrame, cat_col: str, amount_col: str, title: str = "Amount by category") -> go.Figure:
    t = df[[cat_col, amount_col]].dropna().copy()
    t[amount_col] = pd.to_numeric(t[amount_col], errors="coerce")
    t = t.dropna()
    fig = go.Figure(
        go.Box(
            x=t[cat_col].astype(str),
            y=t[amount_col],
            marker=dict(color=C_PRIMARY),
            boxmean=True,
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Amount ($)")
    return _finalize(fig, title)


def plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str,
    name2: str,
    title: str = "Comparison",
) -> go.Figure:
    a = df1.sort_values("ds").copy()
    b = df2.sort_values("ds").copy()
    a["ds"] = pd.to_datetime(a["ds"])
    b["ds"] = pd.to_datetime(b["ds"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a["ds"],
            y=a["y"],
            name=name1,
            mode="lines",
            line=dict(color=C_PRIMARY, width=2.5),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=b["ds"],
            y=b["y"],
            name=name2,
            mode="lines",
            line=dict(color=C_SECONDARY, width=2.5),
            hovertemplate="%{x|%Y-%m}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$")
    return _finalize(fig, title)


def plot_monthly_bars(df: pd.DataFrame, title: str = "Monthly amounts") -> go.Figure:
    d = df.sort_values("ds").copy()
    d["ds"] = pd.to_datetime(d["ds"])
    mean_y = float(d["y"].mean())
    colors = [C_SUCCESS if v >= mean_y else C_DANGER for v in d["y"].tolist()]
    fig = go.Figure(
        go.Bar(
            x=d["ds"].dt.strftime("%Y-%m"),
            y=d["y"],
            marker=dict(color=colors),
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=mean_y, line_dash="dash", line_color="#64748B", annotation_text=f"Mean ${mean_y:,.0f}")
    fig.update_layout(title=title, xaxis_tickangle=-45)
    fig.update_yaxes(tickprefix="$")
    return _finalize(fig)
