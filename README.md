# 🛒 E-Commerce Sales & Customer Analytics with AI

A full-stack Python data analytics project that covers:

- **Data Generation** – synthetic but realistic e-commerce dataset (5 000+ orders)
- **Data Cleaning** – handles duplicates, missing values, negative revenue, future dates
- **Analytics** – KPIs, time-series, category / region / channel breakdowns, RFM
- **AI Prediction** – Random Forest, Gradient Boosting & Linear Regression sales forecast
- **Streamlit UI** – interactive multi-page dashboard with Plotly charts

---

## Project Structure

```
ecommerce_analytics/
├── app.py               ← Streamlit dashboard (run this)
├── data_generator.py    ← Synthetic dataset creator
├── data_cleaner.py      ← Data cleaning & quality report
├── analytics.py         ← All aggregation & KPI functions
├── predictor.py         ← AI/ML sales prediction
├── charts.py            ← Plotly chart builders
├── requirements.txt     ← Python dependencies
└── data/
    └── ecommerce_sales.csv   ← Generated on first run
```

---

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The app auto-generates the dataset on first launch.

---

## Dashboard Pages

| Page | What's inside |
|------|--------------|
| 📊 Overview | Revenue KPIs, trend chart, category & order-status pie |
| 🔍 Data Quality | Missing values, cleaning steps, raw vs clean data preview |
| 📦 Sales Analysis | Trends, category bars, top products, discount impact, returns |
| 🗺️ Regional & Channel | Revenue by region, channel, payment method |
| 👥 Customer Analytics | Age segments, top customers, RFM segmentation |
| 🤖 AI Prediction | Model comparison, actual vs predicted, 6-month forecast |
| 💡 Business Insights | Data-driven recommendations & decision matrix |

---

## AI Models

| Model | Notes |
|-------|-------|
| **Random Forest** | Primary model; tree ensemble with lag + rolling features |
| **Gradient Boosting** | Sequential boosting; often second-best |
| **Linear Regression** | Baseline with StandardScaler pipeline |

Features: lag-1/2/3/6, rolling mean/std (3m, 6m), month, quarter, year trend, Q4 flag, cyclical month encoding.

---

## Tech Stack

- **Pandas / NumPy** – data wrangling
- **Scikit-Learn** – ML models, TimeSeriesSplit cross-validation
- **Plotly** – interactive charts
- **Streamlit** – web UI
- **Faker** – synthetic data
