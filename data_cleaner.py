"""
data_cleaner.py
---------------
Performs all data quality checks, cleaning, and preprocessing steps.
Returns a CleaningReport dataclass alongside the cleaned DataFrame so
the Streamlit UI can display exactly what was fixed.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
import datetime


@dataclass
class CleaningReport:
    original_shape: tuple
    final_shape: tuple
    issues: List[Dict[str, Any]] = field(default_factory=list)
    missing_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    dtype_changes: List[str] = field(default_factory=list)

    def add_issue(self, step: str, description: str, rows_affected: int):
        self.issues.append({
            "Step": step,
            "Description": description,
            "Rows Affected": rows_affected,
        })

    @property
    def issues_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues)

    @property
    def total_rows_removed(self) -> int:
        return self.original_shape[0] - self.final_shape[0]


def _missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    summary = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            summary.append({
                "Column": col,
                "Missing Count": n_missing,
                "Missing %": round(n_missing / total * 100, 2),
                "dtype": str(df[col].dtype),
            })
    return pd.DataFrame(summary) if summary else pd.DataFrame(
        columns=["Column", "Missing Count", "Missing %", "dtype"]
    )


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Run the full cleaning pipeline.
    Returns (cleaned_df, CleaningReport).
    """
    report = CleaningReport(original_shape=df.shape, final_shape=df.shape)
    df = df.copy()

    # ── 0. Capture missing-value snapshot BEFORE cleaning ─────────────────────
    report.missing_summary = _missing_value_summary(df)

    # ── 1. Ensure correct dtypes ──────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        report.dtype_changes.append("order_date → datetime64")

    for col in ["unit_price", "revenue", "discount_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["quantity", "customer_age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 2. Remove exact duplicates ────────────────────────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        report.add_issue("Duplicate Removal", "Exact duplicate rows removed", n_dupes)

    # ── 3. Remove / flag future dates ─────────────────────────────────────────
    today = pd.Timestamp(datetime.date.today())
    future_mask = df["order_date"] > today
    n_future = future_mask.sum()
    if n_future:
        df = df[~future_mask]
        report.add_issue("Future Dates", "Orders with order_date in the future removed", n_future)

    # ── 4. Remove rows with NaT order_date ────────────────────────────────────
    nat_mask = df["order_date"].isna()
    n_nat = nat_mask.sum()
    if n_nat:
        df = df[~nat_mask]
        report.add_issue("Invalid Dates", "Rows with unparseable order_date removed", n_nat)

    # ── 5. Remove negative / zero revenue ─────────────────────────────────────
    neg_mask = df["revenue"] <= 0
    n_neg = neg_mask.sum()
    if n_neg:
        df = df[~neg_mask]
        report.add_issue("Negative Revenue", "Rows with revenue ≤ 0 removed (data-entry errors)", n_neg)

    # ── 6. Remove zero-quantity orders ────────────────────────────────────────
    zero_qty = df["quantity"] == 0
    n_zero = zero_qty.sum()
    if n_zero:
        df = df[~zero_qty]
        report.add_issue("Zero Quantity", "Orders with quantity = 0 removed", n_zero)

    # ── 7. Impute missing customer_age with median ────────────────────────────
    n_age_missing = df["customer_age"].isna().sum()
    if n_age_missing:
        median_age = df["customer_age"].median()
        df["customer_age"] = df["customer_age"].fillna(median_age)
        report.add_issue("Age Imputation", f"Missing customer_age filled with median ({median_age:.0f})", n_age_missing)

    # ── 8. Impute missing region with mode ────────────────────────────────────
    n_region_missing = df["region"].isna().sum()
    if n_region_missing:
        mode_region = df["region"].mode()[0]
        df["region"] = df["region"].fillna(mode_region)
        report.add_issue("Region Imputation", f"Missing region filled with mode ('{mode_region}')", n_region_missing)

    # ── 9. Cap discount_pct at 100 ────────────────────────────────────────────
    bad_discount = (df["discount_pct"] < 0) | (df["discount_pct"] > 100)
    n_bad_disc = bad_discount.sum()
    if n_bad_disc:
        df.loc[bad_discount, "discount_pct"] = df.loc[bad_discount, "discount_pct"].clip(0, 100)
        report.add_issue("Discount Clamp", "discount_pct outside [0,100] clamped", n_bad_disc)

    # ── 10. Keep missing rating as-is (legitimate: not all customers rate) ────
    n_missing_rating = df["rating"].isna().sum()
    if n_missing_rating:
        report.add_issue(
            "Rating NaN Kept",
            f"{n_missing_rating} missing ratings kept (customers did not rate — valid)",
            0,
        )

    # ── 11. Derive helper columns ─────────────────────────────────────────────
    df["year"]          = df["order_date"].dt.year
    df["month"]         = df["order_date"].dt.month
    df["month_name"]    = df["order_date"].dt.strftime("%b")
    df["quarter"]       = df["order_date"].dt.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    df["year_month"]    = df["order_date"].dt.to_period("M").astype(str)
    df["day_of_week"]   = df["order_date"].dt.day_name()
    df["is_returned"]   = (df["order_status"] == "Returned").astype(int)
    df["is_cancelled"]  = (df["order_status"] == "Cancelled").astype(int)
    df["age_group"]     = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18–25", "26–35", "36–45", "46–55", "56+"],
        right=True,
    )

    df.reset_index(drop=True, inplace=True)
    report.final_shape = df.shape

    return df, report


def get_data_quality_metrics(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
    """High-level quality metrics for the dashboard KPI cards."""
    return {
        "original_rows":    raw_df.shape[0],
        "cleaned_rows":     clean_df.shape[0],
        "rows_removed":     raw_df.shape[0] - clean_df.shape[0],
        "pct_retained":     round(clean_df.shape[0] / raw_df.shape[0] * 100, 1),
        "total_missing_raw": raw_df.isna().sum().sum(),
        "total_missing_clean": clean_df[["customer_age", "region", "rating"]].isna().sum().sum(),
        "duplicate_count":  raw_df.duplicated().sum(),
    }
