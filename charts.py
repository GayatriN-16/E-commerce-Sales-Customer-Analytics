"""
charts.py
---------
All Plotly chart builder functions.
Every function returns a plotly Figure object ready for st.plotly_chart().
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE   = px.colors.qualitative.Bold
ACCENT    = "#3B82F6"
WARN      = "#F59E0B"
DANGER    = "#EF4444"
SUCCESS   = "#10B981"

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _apply(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS, title=title, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Revenue Over Time (line + area)
# ─────────────────────────────────────────────────────────────────────────────

def revenue_over_time(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_df["date"], y=monthly_df["total_revenue"],
        mode="lines+markers",
        name="Monthly Revenue",
        line=dict(color=ACCENT, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    # 3-month rolling average
    monthly_df = monthly_df.copy()
    monthly_df["roll3"] = monthly_df["total_revenue"].rolling(3, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=monthly_df["date"], y=monthly_df["roll3"],
        mode="lines",
        name="3-Month Avg",
        line=dict(color=WARN, width=2, dash="dot"),
        hovertemplate="3M Avg: $%{y:,.0f}<extra></extra>",
    ))
    return _apply(fig, "Monthly Revenue Trend", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sales Forecast
# ─────────────────────────────────────────────────────────────────────────────

def sales_forecast_chart(monthly_df: pd.DataFrame, forecast_df: pd.DataFrame,
                         actuals_vs_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=monthly_df["date"], y=monthly_df["total_revenue"],
        name="Historical", mode="lines",
        line=dict(color=ACCENT, width=2),
    ))

    # Model fit on test set
    if len(actuals_vs_pred):
        fig.add_trace(go.Scatter(
            x=actuals_vs_pred["date"], y=actuals_vs_pred["predicted"],
            name="Model Fit", mode="lines+markers",
            line=dict(color=SUCCESS, width=2, dash="dash"),
        ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_revenue"],
        name="AI Forecast", mode="lines+markers",
        line=dict(color=DANGER, width=2.5, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Predicted: $%{y:,.0f}<extra></extra>",
    ))

    # Shade forecast region
    if len(forecast_df):
        fig.add_vrect(
            x0=forecast_df["date"].min(), x1=forecast_df["date"].max(),
            fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0,
            annotation_text="Forecast", annotation_position="top left",
        )

    return _apply(fig, "AI Sales Forecast", height=420)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Revenue by Category (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def category_bar(cat_df: pd.DataFrame) -> go.Figure:
    cat_sorted = cat_df.sort_values("total_revenue")
    fig = px.bar(
        cat_sorted, x="total_revenue", y="category",
        orientation="h",
        color="total_revenue",
        color_continuous_scale="Blues",
        text="revenue_share_pct",
        labels={"total_revenue": "Revenue ($)", "category": "Category"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "Total Revenue by Category", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category comparison (multi-metric grouped bar)
# ─────────────────────────────────────────────────────────────────────────────

def category_comparison(cat_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Avg Order Value ($)", "Avg Rating (out of 5)"])
    fig.add_trace(go.Bar(
        x=cat_df["category"], y=cat_df["avg_order_value"],
        marker_color=PALETTE[:len(cat_df)],
        name="Avg Order Value",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=cat_df["category"], y=cat_df["avg_rating"],
        marker_color=PALETTE[:len(cat_df)],
        name="Avg Rating",
        showlegend=False,
    ), row=1, col=2)
    fig.update_layout(**LAYOUT_DEFAULTS, title="Category Performance Comparison", height=380)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Revenue pie chart
# ─────────────────────────────────────────────────────────────────────────────

def revenue_pie(df: pd.DataFrame, group_col: str, value_col: str = "total_revenue",
                title: str = "") -> go.Figure:
    fig = px.pie(df, names=group_col, values=value_col,
                 color_discrete_sequence=PALETTE, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply(fig, title, height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Region map / bar
# ─────────────────────────────────────────────────────────────────────────────

def region_bar(region_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        region_df, x="region", y="total_revenue",
        color="region", color_discrete_sequence=PALETTE,
        text="revenue_share_pct",
        labels={"total_revenue": "Revenue ($)", "region": "Region"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return _apply(fig, "Revenue by Region", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Channel revenue
# ─────────────────────────────────────────────────────────────────────────────

def channel_bar(channel_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        channel_df, x="channel", y="total_revenue",
        color="channel", color_discrete_sequence=PALETTE,
        text="revenue_share_pct",
        labels={"total_revenue": "Revenue ($)", "channel": "Channel"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return _apply(fig, "Revenue by Sales Channel", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Order status donut
# ─────────────────────────────────────────────────────────────────────────────

def order_status_donut(status_df: pd.DataFrame) -> go.Figure:
    colors = [ACCENT, SUCCESS, WARN, DANGER]
    fig = go.Figure(go.Pie(
        labels=status_df["order_status"],
        values=status_df["count"],
        hole=0.5,
        marker_colors=colors[:len(status_df)],
    ))
    fig.update_traces(textinfo="percent+label")
    return _apply(fig, "Order Status Distribution", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top products bar
# ─────────────────────────────────────────────────────────────────────────────

def top_products_bar(prod_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        prod_df.sort_values("total_revenue"), x="total_revenue", y="product",
        orientation="h", color="category", color_discrete_sequence=PALETTE,
        labels={"total_revenue": "Revenue ($)", "product": "Product"},
        text="total_revenue",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    return _apply(fig, "Top 10 Products by Revenue", height=420)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Monthly Growth Rate
# ─────────────────────────────────────────────────────────────────────────────

def growth_rate_chart(monthly_df: pd.DataFrame) -> go.Figure:
    m = monthly_df.dropna(subset=["revenue_growth_pct"])
    colors = [SUCCESS if v >= 0 else DANGER for v in m["revenue_growth_pct"]]
    fig = go.Figure(go.Bar(
        x=m["date"], y=m["revenue_growth_pct"],
        marker_color=colors,
        name="MoM Growth %",
        hovertemplate="<b>%{x|%b %Y}</b><br>Growth: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#6B7280", line_width=1)
    return _apply(fig, "Month-over-Month Revenue Growth (%)", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Discount impact scatter
# ─────────────────────────────────────────────────────────────────────────────

def discount_impact_chart(disc_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Order Count by Discount Band",
                                        "Avg Order Value by Discount Band"])
    fig.add_trace(go.Bar(
        x=disc_df["discount_band"].astype(str),
        y=disc_df["order_count"],
        marker_color=ACCENT, name="Orders", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=disc_df["discount_band"].astype(str),
        y=disc_df["avg_order_value"],
        marker_color=WARN, name="Avg Order Value", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(**LAYOUT_DEFAULTS, title="Discount Impact on Orders & Revenue", height=380)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. RFM segment bar
# ─────────────────────────────────────────────────────────────────────────────

def rfm_segment_chart(rfm_df: pd.DataFrame) -> go.Figure:
    seg_counts = rfm_df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    SEGMENT_COLORS = {
        "Champions":      SUCCESS,
        "Loyal":          ACCENT,
        "Need Attention": WARN,
        "At-Risk":        DANGER,
    }
    seg_counts["color"] = seg_counts["segment"].map(SEGMENT_COLORS)
    fig = px.bar(
        seg_counts, x="segment", y="count",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        text="count",
        labels={"count": "# Customers", "segment": "RFM Segment"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return _apply(fig, "Customer RFM Segments", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Customer age-group revenue
# ─────────────────────────────────────────────────────────────────────────────

def age_group_chart(age_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        age_df, x="age_group", y="total_revenue",
        color="age_group", color_discrete_sequence=PALETTE,
        text="revenue_share_pct",
        labels={"total_revenue": "Revenue ($)", "age_group": "Age Group"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return _apply(fig, "Revenue by Customer Age Group", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Heatmap – avg daily revenue by month × day-of-week
# ─────────────────────────────────────────────────────────────────────────────

def revenue_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Month: %{y}<br>Day: %{x}<br>Avg Revenue: $%{z:,.0f}<extra></extra>",
    ))
    return _apply(fig, "Average Revenue: Month × Day of Week", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    fi_sorted = fi_df.sort_values("Importance")
    fig = px.bar(
        fi_sorted, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "Feature Importance (AI Model)", height=420)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Model metrics comparison radar
# ─────────────────────────────────────────────────────────────────────────────

def model_metrics_bar(metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    metrics_to_plot = ["MAE", "RMSE", "R²"]
    for i, row in metrics_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics_to_plot,
            y=[row["MAE"], row["RMSE"], row["R²"]],
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(barmode="group")
    return _apply(fig, "Model Comparison: MAE, RMSE, R²", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Actual vs Predicted scatter
# ─────────────────────────────────────────────────────────────────────────────

def actual_vs_predicted(avp_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avp_df["date"], y=avp_df["actual"],
        mode="lines+markers", name="Actual",
        line=dict(color=ACCENT, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=avp_df["date"], y=avp_df["predicted"],
        mode="lines+markers", name="Predicted",
        line=dict(color=DANGER, width=2, dash="dot"),
    ))
    return _apply(fig, "Actual vs Predicted Revenue (Test Set)", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Return rate by category
# ─────────────────────────────────────────────────────────────────────────────

def return_rate_chart(ret_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        ret_df, x="return_rate_pct", y="category",
        orientation="h", color="return_rate_pct",
        color_continuous_scale="Reds",
        text="return_rate_pct",
        labels={"return_rate_pct": "Return Rate (%)", "category": "Category"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "Return Rate by Category", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 19. Payment method pie
# ─────────────────────────────────────────────────────────────────────────────

def payment_pie(pay_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        pay_df, names="payment_method", values="total_revenue",
        color_discrete_sequence=PALETTE, hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    return _apply(fig, "Revenue by Payment Method", height=370)


# ─────────────────────────────────────────────────────────────────────────────
# 20. Quarterly revenue bar
# ─────────────────────────────────────────────────────────────────────────────

def quarterly_revenue_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: quarterly revenue broken down by category."""
    completed = df[df["order_status"] == "Completed"].copy()
    completed["yr_q"] = completed["year"].astype(str) + " " + completed["quarter"]
    pivot = completed.pivot_table(
        values="revenue", index="yr_q", columns="category", aggfunc="sum"
    ).fillna(0)

    fig = go.Figure()
    for i, cat in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            name=cat, x=pivot.index, y=pivot[cat],
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(barmode="stack")
    return _apply(fig, "Quarterly Revenue by Category (Stacked)", height=420)
