"""
predictor.py
------------
AI-based sales prediction using Scikit-Learn.

Models used
-----------
1. Random Forest Regressor  — primary model (monthly revenue forecast)
2. Gradient Boosting Regressor — comparison model
3. Linear Regression        — baseline

Features engineered
-------------------
- Lag features (1-month, 2-month, 3-month, 6-month revenue)
- Rolling mean / std (3-month, 6-month windows)
- Month, quarter, is_holiday_season dummies
- Year trend (integer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ModelMetrics:
    name: str
    mae: float
    rmse: float
    r2: float
    cv_rmse_mean: float
    cv_rmse_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.name,
            "MAE": round(self.mae, 2),
            "RMSE": round(self.rmse, 2),
            "R²": round(self.r2, 4),
            "CV RMSE (mean)": round(self.cv_rmse_mean, 2),
            "CV RMSE (std)": round(self.cv_rmse_std, 2),
        }


@dataclass
class PredictionResult:
    forecast_df: pd.DataFrame           # future months with predicted revenue
    metrics_df: pd.DataFrame            # model comparison table
    best_model_name: str
    feature_importance: Optional[pd.DataFrame]
    actuals_vs_predicted: pd.DataFrame  # on test set


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Given a monthly revenue Series indexed by period string,
    engineer lag + rolling features.
    """
    df = monthly.copy()
    df["revenue"] = df["total_revenue"]

    # Lags
    for lag in [1, 2, 3, 6]:
        df[f"lag_{lag}"] = df["revenue"].shift(lag)

    # Rolling stats
    df["roll3_mean"] = df["revenue"].shift(1).rolling(3).mean()
    df["roll6_mean"] = df["revenue"].shift(1).rolling(6).mean()
    df["roll3_std"]  = df["revenue"].shift(1).rolling(3).std()
    df["roll6_std"]  = df["revenue"].shift(1).rolling(6).std()

    # Calendar features (from date column)
    df["month_num"]         = df["date"].dt.month
    df["quarter"]           = df["date"].dt.quarter
    df["year_trend"]        = df["date"].dt.year - df["date"].dt.year.min()
    df["is_q4"]             = (df["quarter"] == 4).astype(int)
    df["sin_month"]         = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"]         = np.cos(2 * np.pi * df["month_num"] / 12)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6",
    "roll3_mean", "roll6_mean", "roll3_std", "roll6_std",
    "month_num", "quarter", "year_trend", "is_q4",
    "sin_month", "cos_month",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core training + prediction
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(
    df_clean: pd.DataFrame,
    forecast_months: int = 6,
) -> PredictionResult:
    """
    Train three models on monthly revenue, evaluate, and forecast
    `forecast_months` months ahead using the best model.
    """
    from analytics import sales_over_time

    # ── 1. Build monthly time-series ─────────────────────────────────────────
    monthly = sales_over_time(df_clean, freq="ME")
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly = monthly.sort_values("date").reset_index(drop=True)

    feat_df = _build_features(monthly)
    if len(feat_df) < 15:
        raise ValueError("Not enough data to train (need ≥ 15 months of history).")

    X = feat_df[FEATURE_COLS].values
    y = feat_df["revenue"].values

    # Train / test split (last 20% = test, no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── 2. Define models ──────────────────────────────────────────────────────
    models: Dict[str, Any] = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
    }

    tscv = TimeSeriesSplit(n_splits=3)
    all_metrics: List[ModelMetrics] = []
    best_score = np.inf
    best_name  = ""
    best_model = None
    actuals_vs_pred = pd.DataFrame()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=tscv, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores

        all_metrics.append(ModelMetrics(
            name=name, mae=mae, rmse=rmse, r2=r2,
            cv_rmse_mean=cv_rmse.mean(), cv_rmse_std=cv_rmse.std()
        ))

        if rmse < best_score:
            best_score = rmse
            best_name  = name
            best_model = model
            actuals_vs_pred = pd.DataFrame({
                "date":      feat_df["date"].values[split_idx:],
                "actual":    y_test.round(2),
                "predicted": y_pred.round(2),
            })

    # ── 3. Forecast future months ─────────────────────────────────────────────
    last_date   = monthly["date"].max()
    future_rows = []
    history_rev = list(feat_df["revenue"].values)

    for i in range(1, forecast_months + 1):
        future_date = last_date + pd.DateOffset(months=i)

        # Build a single-row feature vector using recent history
        row = _make_forecast_row(history_rev, future_date, feat_df["date"].min())
        X_future = np.array([[row[c] for c in FEATURE_COLS]])
        pred_rev  = float(best_model.predict(X_future)[0])
        pred_rev  = max(pred_rev, 0)

        future_rows.append({"date": future_date, "predicted_revenue": round(pred_rev, 2)})
        history_rev.append(pred_rev)  # use prediction as pseudo-history for next step

    forecast_df = pd.DataFrame(future_rows)

    # ── 4. Feature importance (RF / GB only) ─────────────────────────────────
    fi_df = None
    raw_model = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
    if hasattr(raw_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": raw_model.feature_importances_.round(4),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    metrics_df = pd.DataFrame([m.to_dict() for m in all_metrics])

    return PredictionResult(
        forecast_df=forecast_df,
        metrics_df=metrics_df,
        best_model_name=best_name,
        feature_importance=fi_df,
        actuals_vs_predicted=actuals_vs_pred,
    )


def _make_forecast_row(history: List[float], future_date: pd.Timestamp, min_date: pd.Timestamp) -> Dict:
    """Build a feature dict for one forecast step."""
    n = len(history)
    safe = lambda idx: history[idx] if abs(idx) <= n else np.nan

    row = {
        "lag_1":       safe(-1),
        "lag_2":       safe(-2),
        "lag_3":       safe(-3),
        "lag_6":       safe(-6),
        "roll3_mean":  np.mean(history[-3:]) if n >= 3 else np.mean(history),
        "roll6_mean":  np.mean(history[-6:]) if n >= 6 else np.mean(history),
        "roll3_std":   np.std(history[-3:])  if n >= 3 else 0.0,
        "roll6_std":   np.std(history[-6:])  if n >= 6 else 0.0,
        "month_num":   future_date.month,
        "quarter":     (future_date.month - 1) // 3 + 1,
        "year_trend":  future_date.year - min_date.year,
        "is_q4":       int(future_date.month in [10, 11, 12]),
        "sin_month":   np.sin(2 * np.pi * future_date.month / 12),
        "cos_month":   np.cos(2 * np.pi * future_date.month / 12),
    }
    return row
