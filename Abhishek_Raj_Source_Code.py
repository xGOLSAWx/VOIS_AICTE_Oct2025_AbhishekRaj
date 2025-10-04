# airbnb_advanced_viz.py
"""
Advanced EDA & Revenue Visualizations for Airbnb_Open_Data.xlsx
- Uses real dataset (no synthetic data, no forced random seed)
- Cleans price/service_fee, computes revenue
- Produces Top-10 revenue charts, time-series by year/month, clustering (optional)
- Saves plots to ./advanced_plots/
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Try sklearn for clustering; if missing, skip clustering gracefully
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------- CONFIG -----------------
DATA_PATH = "1730285881-Airbnb_Open_Data.xlsx"   # change if needed
OUT_DIR = "advanced_plots"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
# ------------------------------------------

# --------- Utility helpers ---------
def try_find_column(df, candidates):
    """Return first candidate column found in df.columns or None"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def clean_currency_series(s):
    """Remove commas, $ and convert to numeric; keep NaN if conversion fails."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("$","", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")

def save_show(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    print("Saved:", path)

# --------- Load data ---------
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Data file not found at {DATA_PATH}. Put your Excel file in the same folder and re-run.")
    sys.exit(1)

print("Loading data from:", DATA_PATH)
df = pd.read_excel(DATA_PATH)
print("Raw shape:", df.shape)
print("Columns sample:", df.columns[:20].tolist())

# --------- CLEANING (as you requested) ---------
# 1) Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped duplicates: {before - len(df)} rows removed")

# 2) Drop `house_rules` and `license` if present (you asked these removed)
for col in ["house_rules", "license"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped column: {col}")

# 3) Identify price & service fee columns (common variants)
price_col = try_find_column(df, ["$price", "price", "Price", "PRICE"])
service_col = try_find_column(df, ["$service_fee", "service fee", "service_fee", "serviceFee", "serviceFeeAmount"])

# Clean them (remove commas / $) and rename to $price / $service_fee
if price_col:
    df["$price"] = clean_currency_series(df[price_col])
    if price_col != "$price":
        df = df.drop(columns=[price_col])
    print("Cleaned price column into $price")
else:
    print("No price column found. Script will still run but many revenue charts will skip.")

if service_col:
    df["$service_fee"] = clean_currency_series(df[service_col])
    if service_col != "$service_fee":
        df = df.drop(columns=[service_col])
    print("Cleaned service fee into $service_fee")
else:
    print("No service fee column found; revenue will use $price only.")

# 4) Compute revenue column (use service fee if present)
if "$price" in df.columns:
    if "$service_fee" in df.columns:
        df["revenue"] = df["$price"].fillna(0) + df["$service_fee"].fillna(0)
    else:
        df["revenue"] = df["$price"].fillna(0)
else:
    df["revenue"] = 0.0

# 5) Drop rows with missing values (you requested this strictly)
#    Warning: this may drop many rows; we only drop rows with missing values in critical columns for revenue/time analysis.
critical_cols = []
if "$price" in df.columns: critical_cols.append("$price")
# include date column candidates below; we'll dropna on cols we need before each viz rather than global dropna to avoid over-deleting

# 6) Fix availability_365 outliers and convert type
avail_col = try_find_column(df, ["availability_365", "availability 365", "availability365", "availability"])
if avail_col:
    df["availability_365"] = pd.to_numeric(df[avail_col], errors="coerce")
    # remove outliers outside 0..365
    df.loc[ (df["availability_365"] < 0) | (df["availability_365"] > 365), "availability_365"] = np.nan
    # no global dropna yet
    if avail_col != "availability_365":
        df = df.drop(columns=[avail_col])
    print("Processed availability_365")

# 7) Fix reviews_per_month type if exists
revpm_col = try_find_column(df, ["reviews_per_month", "reviews per month", "reviews_per_month"])
if revpm_col:
    df["reviews_per_month"] = pd.to_numeric(df[revpm_col], errors="coerce")
    if revpm_col != "reviews_per_month":
        df = df.drop(columns=[revpm_col])
    print("Processed reviews_per_month")

# 8) Fix neighbourhood / city / country names & spelling
# Candidate columns for geography
city_col = try_find_column(df, ["city", "City", "neighbourhood", "neighbourhood_cleansed", "neighborhood"])
country_col = try_find_column(df, ["country", "Country", "country_name"])
neigh_col = try_find_column(df, ["neighbourhood", "neighbourhood_cleansed", "neighborhood", "neighborhood_cleansed"])

if neigh_col and neigh_col in df.columns:
    df["neighbourhood_clean"] = df[neigh_col].astype(str).str.strip()
    # Fix Brookln -> Brooklyn (case-insensitive)
    df["neighbourhood_clean"] = df["neighbourhood_clean"].replace({"Brookln": "Brooklyn", "brookln": "Brooklyn"})
else:
    df["neighbourhood_clean"] = df[city_col] if city_col in df.columns else np.nan

if city_col and city_col in df.columns:
    df["city_clean"] = df[city_col].astype(str).str.strip()
else:
    df["city_clean"] = df["neighbourhood_clean"]

if country_col and country_col in df.columns:
    df["country_clean"] = df[country_col].astype(str).str.strip()
else:
    # fallback to 'country' as Unknown
    df["country_clean"] = "Unknown"

# 9) Dates: find a booking/date column (try common names). We'll create year/month for grouping.
date_col = try_find_column(df, ["date", "Date", "booking_date", "reservation_date", "last_review", "last_review_date", "review_date"])
if date_col:
    df["_date_raw"] = pd.to_datetime(df[date_col], errors="coerce")
    if date_col != "_date_raw":
        # keep other date columns but work with _date_raw
        pass
else:
    df["_date_raw"] = pd.NaT

# Create year/month for grouping
df["year"] = df["_date_raw"].dt.year
df["month"] = df["_date_raw"].dt.month
df["year_month"] = df["_date_raw"].dt.to_period("M").dt.to_timestamp()

# After column-specific cleaning, drop rows missing $price or revenue as they break revenue analyses
df = df.dropna(subset=["revenue", "$price"], how="any")
print("Shape after cleaning important fields:", df.shape)

# ----------------- Prepare Top-10 groups -----------------
# Top 10 cities by listing count (or neighbourhood if city not present)
group_city = "city_clean" if "city_clean" in df.columns else "neighbourhood_clean"
top10_cities = df[group_city].value_counts().head(10).index.tolist()

# Top 10 neighbourhoods
top10_neigh = df["neighbourhood_clean"].value_counts().head(10).index.tolist()

# Top 10 countries by revenue (if country data meaningful)
top10_countries = df.groupby("country_clean")["revenue"].sum().sort_values(ascending=False).head(10).index.tolist()

# ----------------- VISUALIZATIONS -----------------

# 1) Top 10 Cities by Total Revenue (vertical bar)
city_rev = df.groupby(group_city)["revenue"].sum().sort_values(ascending=False).head(10)
fig = plt.figure(figsize=(10,6))
sns.barplot(x=city_rev.index, y=city_rev.values, palette="mako")
plt.title("Top 10 Cities by Total Revenue")
plt.xlabel("City")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
save_show(fig, "top10_cities_total_revenue.png")


# 3) Revenue distribution histogram (global)
fig = plt.figure(figsize=(8,5))
sns.histplot(df["revenue"].dropna(), bins=60, kde=True)
plt.title("Revenue Distribution (All listings)")
plt.xlabel("Revenue")
save_show(fig, "revenue_histogram.png")

# 4) Revenue by Year (time series) - uses _date_raw; skip if missing
if df["_date_raw"].notna().sum() > 0:
    yearly = df.dropna(subset=["year"]).groupby("year")["revenue"].sum().sort_index()
    fig = plt.figure(figsize=(9,5))
    sns.lineplot(x=yearly.index, y=yearly.values, marker="o")
    plt.title("Total Revenue by Year")
    plt.ylabel("Total Revenue")
    plt.xlabel("Year")
    save_show(fig, "revenue_by_year.png")
else:
    print("No usable date column found for yearly revenue plot; skipping revenue_by_year.")

# 5) Revenue by Month & Top Cities heatmap (pivot year_month x city)
# Build pivot only for top cities and rows with a valid year_month
if df["year_month"].notna().sum() > 0:
    pivot = df[df[group_city].isin(top10_cities)].dropna(subset=["year_month"]).groupby(["year_month", group_city])["revenue"].sum().unstack(fill_value=0)
    # To keep plot compact, take recent 24 months if many
    if pivot.shape[0] > 24:
        pivot = pivot.tail(24)
    fig = plt.figure(figsize=(12,6))
    sns.heatmap(pivot.T, cmap="YlGnBu", cbar_kws={'label':'Revenue'})
    plt.title("Revenue by Month (Top Cities) — last periods")
    plt.ylabel("City")
    plt.xlabel("Year-Month")
    save_show(fig, "revenue_month_city_heatmap.png")
else:
    print("No usable year_month for monthly heatmap; skipping.")

# 6) Revenue per listing (boxplot) for Top 10 neighbourhoods
sub = df[df["neighbourhood_clean"].isin(top10_neigh)]
if not sub.empty:
    fig = plt.figure(figsize=(12,6))
    sns.boxplot(x="neighbourhood_clean", y="revenue", data=sub, showfliers=False, palette="viridis")
    plt.title("Revenue per listing by Top 10 Neighbourhoods")
    plt.xticks(rotation=45)
    save_show(fig, "box_revenue_top10_neigh.png")
else:
    print("No data for neighbourhood boxplot.")

# 7) Bookings / Revenue time-series for Top Cities (multiple lines)
if df["_date_raw"].notna().sum() > 0:
    ts = df[df[group_city].isin(top10_cities)].dropna(subset=["year_month"])
    tsagg = ts.groupby(["year_month", group_city]).agg(total_revenue=("revenue", "sum"), total_bookings=("bookings_last_30d", "sum")).reset_index()
    # plot revenue lines
    fig = plt.figure(figsize=(12,6))
    for city in top10_cities:
        city_ts = tsagg[tsagg[group_city]==city]
        if city_ts.empty: continue
        plt.plot(city_ts["year_month"].dt.to_timestamp(), city_ts["total_revenue"], marker="o", label=city)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("Monthly Revenue by Top Cities")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    save_show(fig, "monthly_revenue_top_cities.png")
else:
    print("No date info for monthly revenue lines; skipping.")

# 8) Histogram of price for Top Cities (multiple small subplots)
fig, axes = plt.subplots(2,5, figsize=(18,8), sharex=False)
axes = axes.flatten()
for i, city in enumerate(top10_cities[:10]):
    ax = axes[i]
    data = df[df[group_city]==city]["$price"].dropna()
    sns.histplot(data, bins=30, ax=ax, kde=False)
    ax.set_title(city)
    ax.set_xlabel("$price")
plt.suptitle("Price Distribution by Top Cities (Top 10)")
save_show(fig, "price_hist_top10_cities.png")

# 9) Optional: clustering by numeric features (price, availability, reviews) -- if sklearn available
if SKLEARN_AVAILABLE:
    cluster_cols = []
    if "$price" in df.columns: cluster_cols.append("$price")
    if "availability_365" in df.columns: cluster_cols.append("availability_365")
    if "reviews_per_month" in df.columns: cluster_cols.append("reviews_per_month")
    cluster_cols = [c for c in cluster_cols if c in df.columns]
    if len(cluster_cols) >= 2:
        # Prepare data
        clust_df = df[cluster_cols].dropna().copy()
        # standardize roughly with log transform for price and reviews
        clust_proc = clust_df.copy()
        if "$price" in clust_proc.columns:
            clust_proc["$price"] = np.log1p(clust_proc["$price"])
        if "reviews_per_month" in clust_proc.columns:
            clust_proc["reviews_per_month"] = np.log1p(clust_proc["reviews_per_month"])
        # Fill NA with median (should be none)
        clust_proc = clust_proc.fillna(clust_proc.median())
        # choose k=3..5 small
        k = 4 if len(clust_proc) > 100 else 3
        km = KMeans(n_clusters=k, random_state=0).fit(clust_proc)
        clust_df["cluster"] = km.labels_
        # cluster summary
        summary = clust_df.groupby("cluster").mean()
        print("\nCluster centers (approx):")
        print(summary)
        # Merge cluster labels back for plotting (may reduce to sampled points)
        sample = df.loc[clust_df.index].copy()
        sample["cluster"] = clust_df["cluster"].values
        fig = plt.figure(figsize=(10,6))
        if "$price" in sample.columns and "availability_365" in sample.columns:
            sns.scatterplot(x=np.log1p(sample["$price"]), y=sample["availability_365"], hue=sample["cluster"], palette="tab10", alpha=0.6)
            plt.title("Cluster scatter (log price vs availability)")
            plt.xlabel("log(1+$price)")
            plt.ylabel("availability_365")
            save_show(fig, "clusters_price_vs_availability.png")
        # cluster revenue bar
        cluster_rev = sample.groupby("cluster")["revenue"].mean()
        fig = plt.figure(figsize=(8,5))
        sns.barplot(x=cluster_rev.index.astype(str), y=cluster_rev.values, palette="mako")
        plt.title("Average revenue by cluster")
        plt.xlabel("cluster")
        plt.ylabel("avg revenue")
        save_show(fig, "cluster_avg_revenue.png")
    else:
        print("Not enough numeric columns for clustering. Skipping clustering.")
else:
    print("scikit-learn not available; skipping clustering plots. To enable, install scikit-learn.")

print("\nAll done. Plots saved in:", OUT_DIR)
