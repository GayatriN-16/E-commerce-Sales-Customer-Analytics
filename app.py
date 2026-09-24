"""
app.py  –  E-Commerce Sales & Customer Analytics Dashboard
==========================================================
Run with:   streamlit run app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

# ── Project modules ────────────────────────────────────────────────────────
from data_generator import load_or_generate
from data_cleaner   import clean, get_data_quality_metrics
from analytics      import (
    compute_kpis,
    sales_over_time,
    monthly_growth,
    sales_by_category,
    top_products,
    sales_by_region,
    sales_by_channel,
    customer_segments,
    rfm_analysis,
    top_customers,
    order_status_summary,
    return_rate_by_category,
    discount_impact,
    sales_by_payment,
    sales_heatmap_data,
)
from charts import (
    revenue_over_time,
    sales_forecast_chart,
    category_bar,
    category_comparison,
    revenue_pie,
    region_bar,
    channel_bar,
    order_status_donut,
    top_products_bar,
    growth_rate_chart,
    discount_impact_chart,
    rfm_segment_chart,
    age_group_chart,
    revenue_heatmap,
    feature_importance_chart,
    model_metrics_bar,
    actual_vs_predicted,
    return_rate_chart,
    payment_pie,
    quarterly_revenue_chart,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    /* KPI cards */
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-value   { font-size: 1.75rem; font-weight: 700; color: #1E293B; }
    .kpi-label   { font-size: 0.82rem; color: #64748B;  margin-top: 4px; }
    .kpi-delta   { font-size: 0.78rem; margin-top: 6px; }
    .kpi-pos     { color: #10B981; }
    .kpi-neg     { color: #EF4444; }

    /* Section headers */
    .section-header {
        font-size: 1.15rem; font-weight: 600;
        color: #1E293B; border-left: 4px solid #3B82F6;
        padding-left: 10px; margin: 28px 0 14px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        color: #F1F5F9;
    }
    section[data-testid="stSidebar"] * { color: #F1F5F9 !important; }

    /* Insight boxes */
    .insight-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #1E3A5F;
    }
    .warn-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #78350F;
    }

    /* Expander */
    .streamlit-expanderHeader { font-weight: 600 !important; }

    /* Table tweaks */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating & loading dataset …")
def load_data():
    raw_df = load_or_generate("data/ecommerce_sales.csv", n_orders=5000)
    clean_df, report = clean(raw_df)
    return raw_df, clean_df, report


@st.cache_data(show_spinner="Running AI prediction models …")
def run_predictions(_clean_df: pd.DataFrame, forecast_months: int):
    from predictor import train_and_predict
    return train_and_predict(_clean_df, forecast_months=forecast_months)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛒 E-Commerce Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🔍 Data Quality",
            "📦 Sales Analysis",
            "🗺️ Regional & Channel",
            "👥 Customer Analytics",
            "🤖 AI Prediction",
            "💡 Business Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Filters")

    raw_df, clean_df, report = load_data()

    # Date range filter
    min_date = clean_df["order_date"].min().date()
    max_date = clean_df["order_date"].max().date()
    date_range = st.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Category filter
    all_cats = sorted(clean_df["category"].unique())
    sel_cats = st.multiselect("Categories", all_cats, default=all_cats)

    # Region filter
    all_regions = sorted(clean_df["region"].dropna().unique())
    sel_regions = st.multiselect("Regions", all_regions, default=all_regions)

    # Channel filter
    all_channels = sorted(clean_df["channel"].unique())
    sel_channels = st.multiselect("Channels", all_channels, default=all_channels)

    st.markdown("---")
    forecast_months = st.slider("🔮 Forecast Horizon (months)", 3, 12, 6)

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · Scikit-Learn")


# ── Apply filters ──────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d = pd.Timestamp(min_date)
    end_d   = pd.Timestamp(max_date)

df = clean_df.copy()
df = df[(df["order_date"] >= start_d) & (df["order_date"] <= end_d)]
if sel_cats:
    df = df[df["category"].isin(sel_cats)]
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_channels:
    df = df[df["channel"].isin(sel_channels)]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – KPI card
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, delta: str = "", delta_pos: bool = True):
    delta_class = "kpi-pos" if delta_pos else "kpi-neg"
    delta_html  = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def warn(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def fmt_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.2f}"


def fmt_num(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 E-Commerce Sales Dashboard")
    st.caption(f"Showing **{len(df):,}** orders · {start_d.date()} → {end_d.date()}")

    kpis = compute_kpis(df)
    monthly = monthly_growth(df)
    last_growth = monthly["revenue_growth_pct"].iloc[-1] if len(monthly) else 0

    # ── Row 1: Revenue KPIs ────────────────────────────────────────────────
    section("Revenue & Orders")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Revenue",      fmt_currency(kpis["total_revenue"]),
                       f"{last_growth:+.1f}% MoM", last_growth >= 0)
    with c2: kpi_card("Completed Revenue",  fmt_currency(kpis["completed_revenue"]))
    with c3: kpi_card("Total Orders",       fmt_num(kpis["total_orders"]))
    with c4: kpi_card("Avg Order Value",    fmt_currency(kpis["avg_order_value"]))

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5: kpi_card("Unique Customers",  fmt_num(kpis["unique_customers"]))
    with c6: kpi_card("Units Sold",        fmt_num(kpis["total_units_sold"]))
    with c7: kpi_card("Return Rate",       f"{kpis['return_rate_pct']}%",
                       "Target < 8%", kpis["return_rate_pct"] < 8)
    with c8: kpi_card("Avg Rating",        f"⭐ {kpis['avg_rating']}")

    # ── Revenue trend ──────────────────────────────────────────────────────
    section("Revenue Trend")
    monthly_ts = sales_over_time(df, "ME")
    st.plotly_chart(revenue_over_time(monthly_ts), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        section("Revenue by Category")
        cat_df = sales_by_category(df)
        st.plotly_chart(revenue_pie(cat_df, "category", title="Category Share"), use_container_width=True)
    with col_r:
        section("Order Status")
        stat_df = order_status_summary(df)
        st.plotly_chart(order_status_donut(stat_df), use_container_width=True)

    section("Quarterly Revenue Breakdown")
    st.plotly_chart(quarterly_revenue_chart(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Data Quality":
    st.title("🔍 Data Quality Report")
    st.caption("Full audit of the raw dataset, issues detected, and cleaning steps taken.")

    qm = get_data_quality_metrics(raw_df, clean_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Original Rows",   fmt_num(qm["original_rows"]))
    with c2: kpi_card("Cleaned Rows",    fmt_num(qm["cleaned_rows"]))
    with c3: kpi_card("Rows Removed",    fmt_num(qm["rows_removed"]),
                       f"{100 - qm['pct_retained']:.1f}% of dataset", False)
    with c4: kpi_card("Retained",        f"{qm['pct_retained']}%", "Data Integrity", True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a: kpi_card("Raw Missing Cells",   fmt_num(qm["total_missing_raw"]))
    with col_b: kpi_card("Clean Missing Cells", fmt_num(qm["total_missing_clean"]))
    with col_c: kpi_card("Duplicates Found",    fmt_num(qm["duplicate_count"]))

    # Missing values before cleaning
    section("Missing Values — Raw Dataset")
    if len(report.missing_summary):
        st.dataframe(
            report.missing_summary.style
                .background_gradient(subset=["Missing %"], cmap="OrRd")
                .format({"Missing %": "{:.2f}%"}),
            use_container_width=True,
        )
    else:
        st.success("No missing values found in the raw dataset.")

    # Cleaning steps
    section("Cleaning Steps Applied")
    if len(report.issues_df):
        st.dataframe(
            report.issues_df.style
                .set_properties(**{"background-color": "#F8FAFC"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No issues found.")

    # Raw vs clean sample
    section("Raw Data Preview (first 200 rows)")
    with st.expander("Show raw data", expanded=False):
        st.dataframe(raw_df.head(200), use_container_width=True)

    section("Cleaned Data Preview (first 200 rows)")
    with st.expander("Show cleaned data", expanded=True):
        st.dataframe(clean_df.head(200), use_container_width=True)

    # Dtype info
    section("Column Data Types (cleaned)")
    dtype_df = pd.DataFrame({
        "Column": clean_df.columns,
        "dtype":  [str(t) for t in clean_df.dtypes],
        "Non-Null Count": clean_df.count().values,
        "Null Count": clean_df.isna().sum().values,
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    insight(
        "All future-dated orders, negative-revenue entries, and zero-quantity rows "
        "were removed. Missing ages and regions were imputed with median/mode values "
        "to preserve statistical representativeness."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SALES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Sales Analysis":
    st.title("📦 Sales Analysis")

    tabs = st.tabs(["📈 Trends", "📦 Categories", "🏆 Top Products",
                    "💸 Discounts", "🔁 Returns"])

    # ── Tab 1: Trends ──────────────────────────────────────────────────────
    with tabs[0]:
        section("Monthly Revenue & Growth")
        monthly = monthly_growth(df)
        st.plotly_chart(revenue_over_time(sales_over_time(df, "ME")), use_container_width=True)
        st.plotly_chart(growth_rate_chart(monthly), use_container_width=True)

        section("Revenue Heatmap (Month × Day of Week)")
        pivot = sales_heatmap_data(df)
        if not pivot.empty:
            st.plotly_chart(revenue_heatmap(pivot), use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")

    # ── Tab 2: Categories ──────────────────────────────────────────────────
    with tabs[1]:
        cat_df = sales_by_category(df)

        section("Revenue by Category")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(category_bar(cat_df), use_container_width=True)
        with col2:
            st.plotly_chart(revenue_pie(cat_df, "category", title="Category Share"), use_container_width=True)

        section("Category Performance Comparison")
        st.plotly_chart(category_comparison(cat_df), use_container_width=True)

        section("Full Category Summary Table")
        display_cat = cat_df.copy()
        display_cat["total_revenue"] = display_cat["total_revenue"].apply(fmt_currency)
        display_cat["avg_order_value"] = display_cat["avg_order_value"].apply(fmt_currency)
        st.dataframe(display_cat, use_container_width=True, hide_index=True)

        # Best and worst
        best_cat  = cat_df.iloc[0]["category"]
        worst_cat = cat_df.iloc[-1]["category"]
        insight(f"**{best_cat}** generates the highest revenue — prime candidate for upsell campaigns.")
        warn(f"**{worst_cat}** has the lowest revenue. Review pricing, visibility, or assortment.")

    # ── Tab 3: Top Products ────────────────────────────────────────────────
    with tabs[2]:
        prod_df = top_products(df, 10)
        section("Top 10 Products by Revenue")
        st.plotly_chart(top_products_bar(prod_df), use_container_width=True)

        section("Top Products Table")
        disp_prod = prod_df.copy()
        disp_prod["total_revenue"] = disp_prod["total_revenue"].apply(fmt_currency)
        disp_prod["avg_price"]     = disp_prod["avg_price"].apply(fmt_currency)
        st.dataframe(disp_prod, use_container_width=True, hide_index=True)

    # ── Tab 4: Discounts ───────────────────────────────────────────────────
    with tabs[3]:
        disc_df = discount_impact(df)
        section("Discount Impact Analysis")
        st.plotly_chart(discount_impact_chart(disc_df), use_container_width=True)
        st.dataframe(disc_df, use_container_width=True, hide_index=True)

        # Insight: find discount band with highest avg order value
        best_disc = disc_df.loc[disc_df["avg_order_value"].idxmax(), "discount_band"]
        insight(
            f"Orders with **{best_disc}** discount band show the highest average order value. "
            "Consider targeted promotions in this range to maximise both conversion and revenue."
        )

    # ── Tab 5: Returns ─────────────────────────────────────────────────────
    with tabs[4]:
        ret_df = return_rate_by_category(df)
        section("Return Rate by Category")
        st.plotly_chart(return_rate_chart(ret_df), use_container_width=True)
        st.dataframe(ret_df, use_container_width=True, hide_index=True)

        high_ret = ret_df.iloc[0]
        warn(
            f"**{high_ret['category']}** has the highest return rate "
            f"({high_ret['return_rate_pct']:.1f}%). "
            "Investigate product descriptions, sizing guides, and packaging quality."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGIONAL & CHANNEL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Regional & Channel":
    st.title("🗺️ Regional & Channel Analysis")

    section("Revenue by Region")
    region_df = sales_by_region(df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(region_bar(region_df), use_container_width=True)
    with col2:
        st.plotly_chart(revenue_pie(region_df, "region", title="Regional Share"), use_container_width=True)

    section("Region Summary Table")
    st.dataframe(region_df.style.format({
        "total_revenue":   "${:,.2f}",
        "avg_order_value": "${:,.2f}",
        "return_rate_pct": "{:.2f}%",
        "revenue_share_pct": "{:.2f}%",
    }), use_container_width=True, hide_index=True)

    st.markdown("---")

    section("Revenue by Sales Channel")
    channel_df = sales_by_channel(df)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(channel_bar(channel_df), use_container_width=True)
    with col4:
        st.plotly_chart(revenue_pie(channel_df, "channel", title="Channel Share"), use_container_width=True)

    section("Channel Summary Table")
    st.dataframe(channel_df.style.format({
        "total_revenue":   "${:,.2f}",
        "avg_order_value": "${:,.2f}",
        "avg_discount":    "{:.2f}%",
        "revenue_share_pct": "{:.2f}%",
    }), use_container_width=True, hide_index=True)

    st.markdown("---")

    section("Revenue by Payment Method")
    pay_df = sales_by_payment(df)
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(payment_pie(pay_df), use_container_width=True)
    with col6:
        st.dataframe(pay_df.style.format({
            "total_revenue":     "${:,.2f}",
            "avg_order_value":   "${:,.2f}",
            "revenue_share_pct": "{:.2f}%",
        }), use_container_width=True, hide_index=True)

    top_region  = region_df.iloc[0]["region"]
    top_channel = channel_df.iloc[0]["channel"]
    insight(f"**{top_region}** is the top revenue region. Consider additional warehouse or logistics investment here.")
    insight(f"**{top_channel}** drives the most sales. Prioritise UX improvements and targeted ads on this channel.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Customer Analytics":
    st.title("👥 Customer Analytics")

    tabs = st.tabs(["🎯 Segments", "👑 Top Customers", "📊 Age Groups", "🔄 RFM Analysis"])

    with tabs[0]:
        section("Customer Segments by Age Group")
        age_df = customer_segments(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(age_group_chart(age_df), use_container_width=True)
        with col2:
            st.plotly_chart(revenue_pie(age_df, "age_group", title="Revenue by Age Group"), use_container_width=True)
        section("Age Group Summary")
        st.dataframe(age_df.style.format({
            "total_revenue":     "${:,.2f}",
            "avg_order_value":   "${:,.2f}",
            "revenue_share_pct": "{:.2f}%",
        }), use_container_width=True, hide_index=True)

    with tabs[1]:
        section("Top 10 Customers by Total Spend")
        top_c = top_customers(df, 10)
        st.dataframe(top_c.style.format({
            "total_spend": "${:,.2f}",
            "avg_order":   "${:,.2f}",
            "avg_rating":  "{:.2f}",
        }).background_gradient(subset=["total_spend"], cmap="Blues"),
        use_container_width=True, hide_index=True)

        kpis = compute_kpis(df)
        insight(
            f"Top 10 customers represent high lifetime value. "
            f"Average orders per customer: **{kpis['avg_orders_per_customer']}**. "
            "Consider a loyalty programme to retain high-value customers."
        )

    with tabs[2]:
        section("Revenue & Orders by Age Group")
        age_df = customer_segments(df)
        st.plotly_chart(age_group_chart(age_df), use_container_width=True)
        best_age = age_df.loc[age_df["total_revenue"].idxmax(), "age_group"]
        insight(f"The **{best_age}** age group contributes the most revenue. Tailor marketing creatives for this cohort.")

    with tabs[3]:
        section("RFM Customer Segmentation")
        st.markdown("""
        RFM scores customers on three dimensions:
        - **Recency** – how recently they purchased
        - **Frequency** – how often they purchase
        - **Monetary** – how much they spend
        """)
        with st.spinner("Computing RFM …"):
            rfm_df = rfm_analysis(df)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rfm_segment_chart(rfm_df), use_container_width=True)
        with col2:
            seg_summary = rfm_df.groupby("segment", observed=True).agg(
                customers=("customer_id", "count"),
                avg_monetary=("monetary", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_recency=("recency", "mean"),
            ).reset_index()
            seg_summary["avg_monetary"]  = seg_summary["avg_monetary"].round(2)
            seg_summary["avg_frequency"] = seg_summary["avg_frequency"].round(2)
            seg_summary["avg_recency"]   = seg_summary["avg_recency"].round(0).astype(int)
            st.dataframe(seg_summary.style.format({
                "avg_monetary": "${:,.2f}",
            }).background_gradient(subset=["avg_monetary"], cmap="Greens"),
            use_container_width=True, hide_index=True)

        champions = rfm_df[rfm_df["segment"] == "Champions"].shape[0]
        at_risk   = rfm_df[rfm_df["segment"] == "At-Risk"].shape[0]
        insight(f"**{champions}** customers are Champions. Reward them with exclusive offers to maintain loyalty.")
        warn(f"**{at_risk}** customers are At-Risk. Launch re-engagement campaigns with personalised incentives.")

        with st.expander("View full RFM table"):
            st.dataframe(rfm_df.style.format({"monetary": "${:,.2f}"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AI Prediction":
    st.title("🤖 AI-Based Sales Prediction")
    st.markdown("""
    Three machine learning models are trained on historical monthly revenue:
    **Random Forest**, **Gradient Boosting**, and **Linear Regression** (baseline).
    The best model (lowest RMSE on the test set) is used for the revenue forecast.
    """)

    monthly_ts = sales_over_time(df, "ME")
    if len(monthly_ts) < 15:
        st.error("⚠️ Not enough historical months in the filtered dataset to train the models. "
                 "Please expand the date range or reduce filters.")
    else:
        with st.spinner("Training models and generating forecast …"):
            result = run_predictions(clean_df, forecast_months)

        # ── Model metrics ──────────────────────────────────────────────────
        section("Model Performance Comparison")
        st.markdown(f"🏆 **Best Model:** `{result.best_model_name}` (lowest RMSE on test set)")
        st.dataframe(
            result.metrics_df.style.highlight_min(subset=["RMSE", "MAE"], color="#DCFCE7")
                                   .highlight_max(subset=["R²"],          color="#DCFCE7")
                                   .format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "R²": "{:.4f}",
                                            "CV RMSE (mean)": "${:,.2f}", "CV RMSE (std)": "${:,.2f}"}),
            use_container_width=True, hide_index=True,
        )
        st.plotly_chart(model_metrics_bar(result.metrics_df), use_container_width=True)

        # ── Actual vs Predicted ────────────────────────────────────────────
        section("Actual vs Predicted Revenue (Test Set)")
        st.plotly_chart(actual_vs_predicted(result.actuals_vs_predicted), use_container_width=True)

        # ── Full forecast chart ────────────────────────────────────────────
        section(f"Revenue Forecast — Next {forecast_months} Months")
        st.plotly_chart(
            sales_forecast_chart(monthly_ts, result.forecast_df, result.actuals_vs_predicted),
            use_container_width=True,
        )

        # Forecast table
        fc = result.forecast_df.copy()
        fc["Month"] = fc["date"].dt.strftime("%B %Y")
        fc["Predicted Revenue"] = fc["predicted_revenue"].apply(fmt_currency)
        st.dataframe(fc[["Month", "Predicted Revenue"]], use_container_width=True, hide_index=True)

        total_forecast = result.forecast_df["predicted_revenue"].sum()
        insight(
            f"The AI model forecasts **{fmt_currency(total_forecast)}** in total revenue "
            f"over the next {forecast_months} months. Use this to plan inventory, staffing, and ad budgets."
        )

        # ── Feature importance ─────────────────────────────────────────────
        if result.feature_importance is not None:
            section("Feature Importance")
            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(feature_importance_chart(result.feature_importance), use_container_width=True)
            with col2:
                st.dataframe(
                    result.feature_importance.style.bar(subset=["Importance"], color="#3B82F6"),
                    use_container_width=True, hide_index=True,
                )
            top_feat = result.feature_importance.iloc[0]["Feature"]
            insight(f"**{top_feat}** is the most predictive feature. "
                    "Lag features confirm that recent sales momentum strongly predicts future revenue.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡 Business Insights":
    st.title("💡 Business Insights & Recommendations")
    st.caption("Data-driven recommendations derived from the full analytics pipeline.")

    kpis     = compute_kpis(df)
    cat_df   = sales_by_category(df)
    region_df= sales_by_region(df)
    ch_df    = sales_by_channel(df)
    ret_df   = return_rate_by_category(df)
    disc_df  = discount_impact(df)
    monthly  = monthly_growth(df)

    # ── KPI snapshot ──────────────────────────────────────────────────────
    section("📌 Performance Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Revenue",    fmt_currency(kpis["total_revenue"]))
    with c2: st.metric("Total Orders",     fmt_num(kpis["total_orders"]))
    with c3: st.metric("Avg Order Value",  fmt_currency(kpis["avg_order_value"]))
    with c4: st.metric("Return Rate",      f"{kpis['return_rate_pct']}%")
    with c5: st.metric("Avg Rating",       f"⭐ {kpis['avg_rating']}")

    st.markdown("---")

    # ── Revenue insights ──────────────────────────────────────────────────
    section("📈 Revenue Insights")
    best_cat    = cat_df.iloc[0]["category"]
    worst_cat   = cat_df.iloc[-1]["category"]
    best_region = region_df.iloc[0]["region"]
    best_ch     = ch_df.iloc[0]["channel"]

    insight(f"**Top category: {best_cat}** — allocate more inventory and run targeted promotions.")
    insight(f"**Top region: {best_region}** — invest in warehousing and faster delivery options.")
    insight(f"**Top channel: {best_ch}** — prioritise UX improvements and channel-specific campaigns.")

    avg_growth = monthly["revenue_growth_pct"].mean()
    if avg_growth > 0:
        insight(f"Average month-over-month revenue growth is **{avg_growth:.1f}%** — business is on a positive trajectory.")
    else:
        warn(f"Average MoM growth is **{avg_growth:.1f}%** — review pricing strategy and marketing spend.")

    st.markdown("---")

    # ── Return rate insights ───────────────────────────────────────────────
    section("🔁 Returns & Cancellations")
    high_ret = ret_df.iloc[0]
    warn(
        f"**{high_ret['category']}** has the highest return rate ({high_ret['return_rate_pct']:.1f}%). "
        "Review product descriptions, images, and sizing charts to reduce returns."
    )
    if kpis["cancellation_rate_pct"] > 10:
        warn(
            f"Cancellation rate is **{kpis['cancellation_rate_pct']}%** — above 10% threshold. "
            "Improve checkout experience, payment options, and delivery time estimates."
        )
    else:
        insight(f"Cancellation rate is under control at **{kpis['cancellation_rate_pct']}%**.")

    st.markdown("---")

    # ── Discount insights ──────────────────────────────────────────────────
    section("💸 Discount Strategy")
    best_disc_band = disc_df.loc[disc_df["avg_order_value"].idxmax(), "discount_band"]
    insight(
        f"Discount band **{best_disc_band}** yields the highest average order value. "
        "This sweet-spot discount level maximises revenue per order without excessive margin erosion."
    )
    high_disc = disc_df[disc_df["discount_band"].astype(str) == ">20%"]
    if len(high_disc) and high_disc.iloc[0]["avg_order_value"] < kpis["avg_order_value"]:
        warn("Discounts above 20% result in below-average order values. Limit deep discounts to clearance campaigns only.")

    st.markdown("---")

    # ── Customer insights ──────────────────────────────────────────────────
    section("👥 Customer Strategy")
    insight(
        f"With **{kpis['unique_customers']:,}** unique customers and "
        f"**{kpis['avg_orders_per_customer']}** average orders per customer, "
        "there is significant potential for increasing purchase frequency through email campaigns and loyalty rewards."
    )
    insight("Introduce a tiered loyalty programme: Silver (2+ orders), Gold (5+), Platinum (10+).")
    insight("Use AI-predicted high-revenue months (Q4) to front-load inventory and launch early-access sales.")

    st.markdown("---")

    # ── Full summary table ─────────────────────────────────────────────────
    section("📋 Category-Level Decision Matrix")
    decision_df = cat_df[["category", "total_revenue", "avg_order_value", "avg_discount", "avg_rating"]].copy()
    decision_df["Action"] = decision_df.apply(lambda r: (
        "🚀 Scale Up"       if r["total_revenue"] >= cat_df["total_revenue"].quantile(0.75) else
        "⚠️ Review"         if r["total_revenue"] <= cat_df["total_revenue"].quantile(0.25) else
        "✅ Maintain"
    ), axis=1)
    st.dataframe(
        decision_df.style.format({
            "total_revenue":   "${:,.2f}",
            "avg_order_value": "${:,.2f}",
            "avg_discount":    "{:.2f}%",
            "avg_rating":      "{:.2f}",
        }).background_gradient(subset=["total_revenue"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )
