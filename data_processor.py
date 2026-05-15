"""
FlowCast — ingest CSV/Excel, normalize columns, aggregate monthly metrics.
"""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Literal, Tuple, Union

import numpy as np
import pandas as pd

FileArg = Union[str, io.BytesIO, BinaryIO]

DATE_CANDIDATES = (
    "date",
    "Date",
    "DATE",
    "transaction_date",
    "Trans Date",
    "posting_date",
    "month",
    "Month",
    "period",
)
AMOUNT_CANDIDATES = (
    "amount",
    "Amount",
    "AMOUNT",
    "debit",
    "credit",
    "transaction",
    "value",
    "sum",
    "total",
    "spending",
    "revenue",
)
CATEGORY_CANDIDATES = (
    "category",
    "Category",
    "type",
    "Type",
    "description",
    "Description",
    "merchant",
    "name",
    "Name",
    "label",
)
# Prefer explicit debit/credit columns before ambiguous "type"
DRCR_CANDIDATES = ("DrCr", "dr_cr", "debit_credit", "DC", "DRCR", "Dr_CR")


def _lower_map(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lm = _lower_map(df)
    for name in candidates:
        if name in df.columns:
            return name
        k = name.lower()
        if k in lm:
            return lm[k]
    return None


def _guess_drcr_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Pick a column whose values look like debit/credit codes."""
    tokens = {"db", "dr", "debit", "cr", "credit"}
    best_col = None
    best_score = 0.0
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c].dropna().astype(str).str.strip().str.lower()
        if s.empty:
            continue
        score = float(s.isin(tokens).mean())
        if score > 0.55 and score > best_score:
            best_score = score
            best_col = c
    return best_col


def _normalize_drcr(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=object)
    debit_mask = s.isin(["db", "dr", "debit"])
    credit_mask = s.isin(["cr", "credit"])
    out.loc[debit_mask] = "Db"
    out.loc[credit_mask] = "Cr"
    return out


def _read_table(file: FileArg, filename: str | None) -> pd.DataFrame:
    name = (filename or "").lower()
    if isinstance(file, str):
        name = file.lower()
    ext = os.path.splitext(name)[1].lower()

    if ext in (".xlsx", ".xls"):
        eng = "openpyxl" if ext == ".xlsx" else None
        try:
            return pd.read_excel(file, engine=eng)
        except Exception as e1:
            try:
                return pd.read_excel(file, engine="openpyxl")
            except Exception as e2:
                raise ValueError(f"Could not read Excel: {e1}; {e2}") from e1

    if ext == ".csv" or ext == "":
        try:
            return pd.read_csv(file)
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}") from e

    try:
        return pd.read_csv(file)
    except Exception:
        return pd.read_excel(file, engine="openpyxl")


def _aggregate_year_month(df: pd.DataFrame, ycol: str = "y") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ds", ycol])
    work = df.copy()
    work["_ym"] = work["ds"].dt.to_period("M")
    out = work.groupby("_ym", as_index=False)[ycol].sum()
    out["ds"] = out["_ym"].dt.to_timestamp()
    out = out.drop(columns=["_ym"]).sort_values("ds", ignore_index=True)
    return out


def _build_summary(
    monthly: pd.DataFrame,
    df_raw: pd.DataFrame,
    mode: str,
    *,
    has_drcr: bool,
    monthly_revenue: pd.DataFrame | None,
    monthly_expense: pd.DataFrame | None,
) -> dict[str, Any]:
    y = monthly["y"].to_numpy(dtype=float) if len(monthly) else np.array([])
    ds = monthly["ds"].to_numpy() if len(monthly) else np.array([])

    total = float(np.nansum(y)) if len(y) else 0.0
    mean = float(np.nanmean(y)) if len(y) else 0.0
    median = float(np.nanmedian(y)) if len(y) else 0.0
    std = float(np.nanstd(y)) if len(y) else 0.0
    min_val = float(np.nanmin(y)) if len(y) else 0.0
    max_val = float(np.nanmax(y)) if len(y) else 0.0

    min_month = max_month = ""
    if len(y):
        imin = int(np.nanargmin(y))
        imax = int(np.nanargmax(y))
        min_month = pd.Timestamp(ds[imin]).strftime("%Y-%m")
        max_month = pd.Timestamp(ds[imax]).strftime("%Y-%m")

    x = np.arange(len(y), dtype=float)
    if len(y) >= 2 and np.all(np.isfinite(y)):
        slope, _ = np.polyfit(x, y, 1)
        trend = "increasing" if slope >= 0 else "decreasing"
    else:
        trend = "increasing"

    if len(y) >= 6:
        first3 = float(np.mean(y[:3]))
        last3 = float(np.mean(y[-3:]))
        if first3 == 0:
            pct_change = float("inf") if last3 != 0 else 0.0
        else:
            pct_change = float((last3 - first3) / first3 * 100.0)
    else:
        pct_change = float("nan")

    amt = df_raw["amount"].to_numpy(dtype=float) if "amount" in df_raw.columns else np.array([])
    total_transactions = int(len(df_raw))
    avg_per_transaction = float(np.mean(np.abs(amt))) if len(amt) else 0.0
    biggest_single_tx = float(np.max(np.abs(amt))) if len(amt) else 0.0

    most_common_category = ""
    if "category" in df_raw.columns and df_raw["category"].notna().any():
        vc = df_raw["category"].astype(str).value_counts()
        if len(vc):
            most_common_category = str(vc.index[0])

    months_above_average = 0
    if len(y) and np.isfinite(mean):
        months_above_average = int(np.sum(y > mean))

    out: dict[str, Any] = {
        "total": total,
        "mean": mean,
        "median": median,
        "std": std,
        "min_val": min_val,
        "max_val": max_val,
        "min_month": min_month,
        "max_month": max_month,
        "trend": trend,
        "pct_change": pct_change,
        "total_transactions": total_transactions,
        "avg_per_transaction": avg_per_transaction,
        "most_common_category": most_common_category,
        "biggest_single_tx": biggest_single_tx,
        "months_above_average": months_above_average,
        "has_drcr": has_drcr,
        "mode": mode,
    }

    if mode == "business" and has_drcr and monthly_revenue is not None and monthly_expense is not None:
        out["total_revenue"] = float(monthly_revenue["revenue"].sum()) if len(monthly_revenue) else 0.0
        out["total_expenses"] = float(monthly_expense["expense"].sum()) if len(monthly_expense) else 0.0
        out["net_total"] = float(out["total_revenue"] - out["total_expenses"])
    else:
        out["total_revenue"] = None
        out["total_expenses"] = None
        out["net_total"] = None

    return out


def load_and_process(
    file: FileArg,
    mode: Literal["personal", "business"] = "personal",
    filename: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """
    Load CSV or Excel, detect columns, build monthly series and a clean row-level frame.

    Returns
    -------
    df_monthly : DataFrame
        Columns ``ds``, ``y`` (forecast-ready).
    df_raw : DataFrame
        Cleaned transactions with ``date``, ``amount``, optional ``category``, ``drcr``.
    summary : dict
        15+ aggregate metrics.
    mode : str
        Echo of ``mode`` argument.
    """
    if mode not in ("personal", "business"):
        raise ValueError("mode must be 'personal' or 'business'")

    try:
        raw = _read_table(file, filename)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read file: {e}") from e

    if raw.empty:
        raise ValueError("File is empty.")

    date_col = _first_existing_column(raw, DATE_CANDIDATES)
    amount_col = _first_existing_column(raw, AMOUNT_CANDIDATES)
    if date_col is None:
        raise ValueError(
            "Could not auto-detect a DATE column. "
            f"Tried: {DATE_CANDIDATES}. Found: {list(raw.columns)}"
        )
    if amount_col is None:
        raise ValueError(
            "Could not auto-detect an AMOUNT column. "
            f"Tried: {AMOUNT_CANDIDATES}. Found: {list(raw.columns)}"
        )

    exclude_for_guess = {date_col, amount_col}
    drcr_col = _first_existing_column(raw, DRCR_CANDIDATES)
    if drcr_col is None:
        guessed = _guess_drcr_column(raw, exclude_for_guess)
        drcr_col = guessed

    cat_col = _first_existing_column(raw, CATEGORY_CANDIDATES)
    if cat_col and drcr_col and cat_col == drcr_col:
        cat_col = None
    if cat_col == amount_col or cat_col == date_col:
        cat_col = None

    use = raw.copy()
    use["_ds"] = pd.to_datetime(use[date_col], errors="coerce")
    use["_amt"] = pd.to_numeric(use[amount_col], errors="coerce")
    use = use.dropna(subset=["_ds", "_amt"])
    if use.empty:
        raise ValueError("No rows with valid date and amount after cleaning.")

    has_drcr = False
    drcr_norm = pd.Series(np.nan, index=use.index, dtype=object)
    if drcr_col is not None and drcr_col in use.columns:
        drcr_norm = _normalize_drcr(use[drcr_col])
        has_drcr = bool(drcr_norm.notna().any())

    df_raw = pd.DataFrame(
        {
            "date": use["_ds"].values,
            "amount": use["_amt"].astype(float).values,
            "category": use[cat_col].astype(str).values if cat_col else "",
            "drcr": drcr_norm.values,
        },
        index=use.index,
    )
    df_raw["category"] = df_raw["category"].replace({"nan": ""})

    monthly_revenue: pd.DataFrame | None = None
    monthly_expense: pd.DataFrame | None = None

    if has_drcr:
        valid_side = df_raw["drcr"].isin(["Db", "Cr"])
        tx = df_raw.loc[valid_side].copy()
        if tx.empty:
            raise ValueError("DrCr column present but no recognizable debit/credit rows.")

        if mode == "personal":
            deb = tx.loc[tx["drcr"] == "Db"].copy()
            if deb.empty:
                raise ValueError("Personal mode with DrCr requires at least one debit row.")
            deb = deb.assign(ds=deb["date"], y=deb["amount"].abs())
            df_monthly = _aggregate_year_month(deb[["ds", "y"]])
        else:
            rev = tx.loc[tx["drcr"] == "Cr"].copy()
            exp = tx.loc[tx["drcr"] == "Db"].copy()
            rev = rev.assign(ds=rev["date"], y=rev["amount"].astype(float))
            exp = exp.assign(ds=exp["date"], y=exp["amount"].abs().astype(float))
            monthly_revenue = _aggregate_year_month(rev[["ds", "y"]]).rename(columns={"y": "revenue"})
            monthly_expense = _aggregate_year_month(exp[["ds", "y"]]).rename(columns={"y": "expense"})
            merged = monthly_revenue.merge(monthly_expense, on="ds", how="outer").fillna(0.0)
            merged["y"] = merged["revenue"] - merged["expense"]
            df_monthly = merged[["ds", "y"]].sort_values("ds", ignore_index=True)
    else:
        tx = df_raw.copy()
        tx["ds"] = tx["date"]
        if mode == "personal":
            tx["y"] = tx["amount"].abs()
        else:
            tx["y"] = tx["amount"].astype(float)
        df_monthly = _aggregate_year_month(tx[["ds", "y"]])

    if df_monthly.empty:
        raise ValueError("Monthly aggregation produced no rows.")

    df_monthly["ds"] = df_monthly["ds"].astype("datetime64[ns]")
    df_monthly["y"] = df_monthly["y"].astype(float)

    summary = _build_summary(
        df_monthly,
        df_raw,
        mode,
        has_drcr=has_drcr,
        monthly_revenue=monthly_revenue,
        monthly_expense=monthly_expense,
    )
    return df_monthly, df_raw, summary, mode
