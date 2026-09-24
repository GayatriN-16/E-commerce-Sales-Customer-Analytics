"""
data_generator.py
-----------------
Generates a realistic synthetic e-commerce sales & customer dataset.
Intentionally injects missing values and outliers so the cleaning
pipeline has something meaningful to handle.
"""

import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty", "Toys", "Automotive"]

PRODUCTS = {
    "Electronics":    [("Wireless Earbuds", 49, 149), ("Smartwatch", 99, 399), ("Laptop Stand", 25, 80),
                       ("USB-C Hub", 20, 70), ("Bluetooth Speaker", 35, 150)],
    "Clothing":       [("Running Shoes", 40, 120), ("Denim Jacket", 45, 130), ("Yoga Pants", 25, 75),
                       ("Polo Shirt", 15, 60), ("Winter Coat", 80, 250)],
    "Home & Kitchen": [("Air Fryer", 60, 200), ("Coffee Maker", 30, 150), ("Knife Set", 20, 100),
                       ("Bed Sheets", 25, 90), ("Blender", 35, 120)],
    "Books":          [("Python Programming", 15, 50), ("Data Science Guide", 18, 55), ("Business Mindset", 12, 40),
                       ("Cook Book", 14, 45), ("Fiction Novel", 8, 25)],
    "Sports":         [("Yoga Mat", 15, 60), ("Resistance Bands", 10, 40), ("Water Bottle", 8, 35),
                       ("Jump Rope", 5, 20), ("Gym Bag", 20, 70)],
    "Beauty":         [("Face Serum", 20, 80), ("Moisturizer", 15, 60), ("Perfume", 30, 120),
                       ("Lip Gloss Set", 10, 35), ("Hair Dryer", 25, 100)],
    "Toys":           [("LEGO Set", 25, 100), ("Remote Car", 20, 80), ("Board Game", 15, 55),
                       ("Puzzle 1000pc", 12, 40), ("Doll Set", 18, 65)],
    "Automotive":     [("Car Phone Mount", 10, 40), ("Dash Cam", 40, 150), ("Seat Covers", 25, 90),
                       ("Tire Inflator", 30, 100), ("Car Vacuum", 20, 70)],
}

REGIONS    = ["North", "South", "East", "West", "Central"]
CHANNELS   = ["Website", "Mobile App", "Marketplace", "In-Store"]
PAYMENT    = ["Credit Card", "Debit Card", "PayPal", "Crypto", "Gift Card"]
STATUSES   = ["Completed", "Returned", "Cancelled", "Pending"]
STATUS_W   = [0.78, 0.08, 0.08, 0.06]


def _seasonal_multiplier(month: int) -> float:
    """Higher sales in Q4 (holiday) and a mid-year bump."""
    base = [0.7, 0.65, 0.8, 0.85, 0.9, 1.0, 1.05, 1.0, 0.95, 1.1, 1.3, 1.6]
    return base[month - 1]


def generate_dataset(n_orders: int = 5000) -> pd.DataFrame:
    records = []
    customer_ids = [f"CUST-{i:05d}" for i in range(1, 1001)]

    for order_idx in range(1, n_orders + 1):
        order_date = fake.date_between(start_date="-2y", end_date="today")
        month      = order_date.month
        multiplier = _seasonal_multiplier(month)

        category = random.choice(CATEGORIES)
        product_name, price_lo, price_hi = random.choice(PRODUCTS[category])

        unit_price  = round(random.uniform(price_lo, price_hi) * multiplier, 2)
        quantity    = max(1, int(np.random.lognormal(mean=0.5, sigma=0.6)))
        discount_pct = round(random.choices([0, 5, 10, 15, 20, 25], weights=[30, 20, 20, 15, 10, 5])[0], 2)
        revenue     = round(unit_price * quantity * (1 - discount_pct / 100), 2)

        customer_id = random.choice(customer_ids)
        region      = random.choice(REGIONS)
        channel     = random.choices(CHANNELS, weights=[40, 30, 20, 10])[0]
        payment     = random.choice(PAYMENT)
        status      = random.choices(STATUSES, weights=STATUS_W)[0]
        rating      = round(random.uniform(1, 5), 1) if random.random() > 0.15 else None
        age         = random.randint(18, 70)

        records.append({
            "order_id":    f"ORD-{order_idx:06d}",
            "order_date":  order_date,
            "customer_id": customer_id,
            "customer_age": age,
            "region":      region,
            "channel":     channel,
            "category":    category,
            "product":     product_name,
            "unit_price":  unit_price,
            "quantity":    quantity,
            "discount_pct": discount_pct,
            "revenue":     revenue,
            "payment_method": payment,
            "order_status": status,
            "rating":      rating,
        })

    df = pd.DataFrame(records)

    # ── Inject realistic data quality issues ──────────────────────────────────
    # 1. Random NaN in non-critical columns
    for col, frac in [("rating", 0.05), ("customer_age", 0.03), ("region", 0.02)]:
        mask = df.sample(frac=frac, random_state=1).index
        df.loc[mask, col] = np.nan

    # 2. Negative revenue outlier (data entry error)
    bad_idx = df.sample(n=20, random_state=2).index
    df.loc[bad_idx, "revenue"] = -df.loc[bad_idx, "revenue"]

    # 3. Zero quantity
    zero_idx = df.sample(n=15, random_state=3).index
    df.loc[zero_idx, "quantity"] = 0

    # 4. Duplicate orders (10 exact dupes)
    dupes = df.sample(n=10, random_state=4)
    df = pd.concat([df, dupes], ignore_index=True)

    # 5. Future dates (impossible)
    future_idx = df.sample(n=8, random_state=5).index
    df.loc[future_idx, "order_date"] = pd.Timestamp("2030-01-01")

    return df


def load_or_generate(path: str = "data/ecommerce_sales.csv", n_orders: int = 5000) -> pd.DataFrame:
    """Load from CSV if it exists; otherwise generate and save."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["order_date"])
    else:
        df = generate_dataset(n_orders)
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/ecommerce_sales.csv", index=False)
    print(f"Dataset saved: {len(df):,} rows × {df.shape[1]} columns")
    print(df.head())
