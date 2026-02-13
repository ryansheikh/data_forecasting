
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Pharmevo Sales Analytics",
    layout="wide"
)

st.title("Pharmevo Sales Analytics Dashboard")
st.caption("Executive-level analytics built on aggregated SQL Server data")

# ==================================================
# CACHE CONTROL (🔥 FIX)
# ==================================================
if st.sidebar.button("🔄 Refresh Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()

def file_signature(path):
    return os.path.getmtime(path)

# ==================================================
# DATA LOADING
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_MAP = {
    "monthly_sales": "cleaned_monthly_sales.csv",
    "monthly_product_sales": "cleaned_monthly_product_sales.csv",
    "top_products": "cleaned_top_products.csv",
    "distributor_performance": "cleaned_distributor_performance.csv",
    "client_type_analysis": "cleaned_client_type_analysis.csv",
    "bonus_discount_monthly": "cleaned_bonus_discount_monthly.csv",
    "dimension_summary": "cleaned_dimension_summary.csv",
    "monthly_client_type_sales": "cleaned_monthly_client_type_sales.csv",
    "price_sensitivity": "cleaned_price_sensitivity.csv",
    "seasonality_monthly_avg": "cleaned_seasonality_monthly_avg.csv",
}

NUMERIC_COLUMNS = {
    "TotalUnits", "TotalBonus", "TotalDiscount", "TotalSales",
    "UnitsSold", "Revenue", "TotalClients",
    "AvgSellingPrice", "AvgMonthlySales"
}

# create file signature so cache auto-refreshes when CSV updates
sig = tuple(file_signature(os.path.join(BASE_DIR, f)) for f in FILE_MAP.values())

@st.cache_data
def load_data(sig):
    data = {}
    for key, fname in FILE_MAP.items():
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            st.error(f"Missing dataset: {fname}")
            st.stop()

        df = pd.read_csv(path)

        for col in df.columns:
            if col in NUMERIC_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "Year" in df.columns and "Month" in df.columns:
            df["MonthStart"] = pd.to_datetime(
                df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
            )

        data[key] = df

    return data

data = load_data(sig)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Executive Overview",
        "Product Performance",
        "Distributor Performance",
        "Client Analysis",
        "Promotion Impact",
        "Seasonality & Cycles",
        "Pricing Analysis",
        "Dimension Drilldown"
    ]
)

# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================
if page == "Executive Overview":
    df = data["monthly_sales"].sort_values("MonthStart").copy()
    df["MoM_Growth"] = df["TotalSales"].pct_change() * 100
    df["Rolling_3M"] = df["TotalSales"].rolling(3).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Latest Sales",
        latest["TotalSales"],
        f"{((latest['TotalSales']-prev['TotalSales'])/prev['TotalSales']*100):.2f}%"
        if prev["TotalSales"] else None
    )
    k2.metric("Latest Units", latest["TotalUnits"])
    k3.metric("Avg Monthly Sales", df["TotalSales"].mean())
    k4.metric("Best Month Sales", df["TotalSales"].max())
    k5.metric("Worst Month Sales", df["TotalSales"].min())

    st.subheader("Sales Trend with Rolling Average")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["TotalSales"], name="Sales"))
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["Rolling_3M"], name="3M Rolling Avg"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Month-on-Month Growth")
    fig2 = px.bar(df, x="MonthStart", y="MoM_Growth")
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# PRODUCT PERFORMANCE
# ==================================================
elif page == "Product Performance":
    top = data["top_products"].sort_values("Revenue", ascending=False)
    total_rev = top["Revenue"].sum()
    top["Share"] = top["Revenue"] / total_rev * 100

    k1, k2 = st.columns(2)
    k1.metric("Top 10 Products Share", f"{top.head(10)['Share'].sum():.2f}%")
    k2.metric("Top 20 Products Share", f"{top.head(20)['Share'].sum():.2f}%")

    st.subheader("Top Products by Revenue")
    fig = px.bar(top.head(20), x="Revenue", y="ProductName", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Revenue Trend")
    mps = data["monthly_product_sales"]
    products = sorted(mps["ProductName"].unique())
    selected = st.multiselect("Select Products", products, default=products[:1])

    if selected:
        fig2 = px.line(
            mps[mps["ProductName"].isin(selected)],
            x="MonthStart",
            y="Revenue",
            color="ProductName",
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# DISTRIBUTOR PERFORMANCE
# ==================================================
elif page == "Distributor Performance":
    dist = data["distributor_performance"].sort_values("Revenue", ascending=False)
    dist["Share"] = dist["Revenue"] / dist["Revenue"].sum() * 100
    dist["CumShare"] = dist["Share"].cumsum()

    st.subheader("Top Distributors")
    fig = px.bar(dist.head(30), x="Revenue", y="DistributorName", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distributor Revenue Concentration")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=dist["Share"], name="Revenue Share"))
    fig2.add_trace(go.Scatter(y=dist["CumShare"], name="Cumulative Share"))
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# CLIENT ANALYSIS
# ==================================================
elif page == "Client Analysis":
    ct = data["client_type_analysis"]

    k1, k2 = st.columns(2)
    k1.metric("Total Client Types", ct["ClientType"].nunique())
    k2.metric(
        "Highest Revenue Client Type",
        ct.sort_values("Revenue", ascending=False).iloc[0]["ClientType"]
    )

    fig = px.pie(ct, names="ClientType", values="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    mct = data["monthly_client_type_sales"]
    fig2 = px.line(mct, x="MonthStart", y="Revenue", color="ClientType", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# PROMOTION IMPACT
# ==================================================
elif page == "Promotion Impact":
    promo = data["bonus_discount_monthly"]
    sales = data["monthly_sales"][["MonthStart", "TotalSales"]]

    merged = promo.merge(sales, on="MonthStart")
    merged["Promo_Total"] = merged["TotalBonus"] + merged["TotalDiscount"]

    k1, k2 = st.columns(2)
    k1.metric("Avg Monthly Promotion", merged["Promo_Total"].mean())
    k2.metric(
        "Promotion to Sales Ratio",
        f"{(merged['Promo_Total'].sum()/merged['TotalSales'].sum())*100:.2f}%"
    )

    fig = px.line(
        merged,
        x="MonthStart",
        y=["TotalBonus", "TotalDiscount"],
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(merged, x="Promo_Total", y="TotalSales")
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# SEASONALITY & CYCLES
# ==================================================
elif page == "Seasonality & Cycles":
    sea = data["seasonality_monthly_avg"]

    fig = px.bar(sea, x="Month", y="AvgMonthlySales")
    st.plotly_chart(fig, use_container_width=True)

    df = data["monthly_sales"]
    heat = df.pivot_table(
        index="Year", columns="Month", values="TotalSales", aggfunc="sum"
    )
    fig2 = px.imshow(heat, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# PRICING ANALYSIS
# ==================================================
elif page == "Pricing Analysis":
    price = data["price_sensitivity"]

    fig = px.bar(
        price.sort_values("AvgSellingPrice", ascending=False).head(30),
        x="AvgSellingPrice",
        y="ProductName",
        orientation="h"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        price,
        x="AvgSellingPrice",
        y="TotalUnits",
        hover_data=["ProductName"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# DIMENSION DRILLDOWN
# ==================================================
elif page == "Dimension Drilldown":
    dim = data["dimension_summary"]

    c1, c2, c3 = st.columns(3)
    with c1:
        distributor = st.selectbox(
            "Distributor", ["All"] + sorted(dim["DistributorName"].unique())
        )
    with c2:
        client_type = st.selectbox(
            "Client Type", ["All"] + sorted(dim["ClientType"].unique())
        )
    with c3:
        team = st.selectbox(
            "Team", ["All"] + sorted(dim["TeamName"].unique())
        )

    df = dim.copy()
    if distributor != "All":
        df = df[df["DistributorName"] == distributor]
    if client_type != "All":
        df = df[df["ClientType"] == client_type]
    if team != "All":
        df = df[df["TeamName"] == team]

    summary = (
        df.groupby("BrickName", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )

    fig = px.bar(summary, x="Revenue", y="BrickName", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
