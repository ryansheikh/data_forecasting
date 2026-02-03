import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Pharma Analytics & Forecasting",
    layout="wide"
)

st.title("Pharma Sales Analytics & Forecasting Dashboard")
st.caption(
    "Complete analytics, ML engineering, unit forecasting, "
    "and product-wise sales planning"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    monthly_sales = pd.read_csv("cleaned_monthly_sales.csv")
    monthly_product_sales = pd.read_csv("cleaned_monthly_product_sales.csv")
    return monthly_sales, monthly_product_sales

monthly_sales, monthly_product_sales = load_data()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Analysis",
        "Units Forecast (Company)",
        "Product Forecast & Sales Planning"
    ]
)

# ==================================================
# EXECUTIVE ANALYSIS (ANALYTICS PART)
# ==================================================
if page == "Executive Analysis":

    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")

    # KPIs
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Units", round(latest["TotalUnits"], 2))
    c2.metric("Latest Sales", round(latest["TotalSales"], 2))
    c3.metric("Avg Monthly Units", round(df["TotalUnits"].mean(), 2))

    st.subheader("Historical Units Trend")

    fig = px.line(
        df,
        x="MonthStart",
        y="TotalUnits",
        markers=True,
        title="Monthly Units Sold (Actual)",
        labels={"MonthStart": "Month", "TotalUnits": "Units"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# UNITS FORECAST (COMPANY LEVEL - ML ENGINEERING)
# ==================================================
elif page == "Units Forecast (Company)":

    st.subheader("Company-Level Units Forecast (ML Based)")

    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")
    df["TotalUnits"] = pd.to_numeric(df["TotalUnits"], errors="coerce").fillna(0)

    ts = df.set_index("MonthStart")["TotalUnits"].asfreq("MS").fillna(0)

    # SARIMA MODEL
    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    horizon = st.slider("Forecast Months", 3, 6, 6)

    forecast = res.get_forecast(steps=horizon).summary_frame()
    forecast_units = forecast["mean"].round(2)

    # Prepare chart data
    hist_df = ts.reset_index()
    hist_df.columns = ["Month", "Actual Units"]

    fc_df = forecast_units.reset_index()
    fc_df.columns = ["Month", "Predicted Units"]

    st.subheader("Actual vs Predicted Units")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["Month"],
        y=hist_df["Actual Units"],
        name="Actual Units",
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["Month"],
        y=fc_df["Predicted Units"],
        name="Predicted Units",
        mode="lines+markers"
    ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Units",
        title="Company-Level Units Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        fc_df.rename(columns={"Predicted Units": "Forecasted Units"}),
        use_container_width=True
    )

# ==================================================
# PRODUCT FORECAST + SALES PLANNING
# ==================================================
elif page == "Product Forecast & Sales Planning":

    st.subheader("Product-Level Units & Sales Forecast")

    # -----------------------------
    # COMPANY UNITS FORECAST
    # -----------------------------
    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")
    ts = df.set_index("MonthStart")["TotalUnits"].asfreq("MS").fillna(0)

    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    horizon = st.slider("Forecast Months ", 3, 6, 6)
    forecast_units = res.get_forecast(steps=horizon).summary_frame()["mean"]

    # -----------------------------
    # PRODUCT SHARE
    # -----------------------------
    monthly_product_sales["UnitsSold"] = pd.to_numeric(
        monthly_product_sales["UnitsSold"], errors="coerce"
    ).fillna(0)

    product_share = (
        monthly_product_sales.groupby("ProductName")["UnitsSold"].sum()
        / monthly_product_sales["UnitsSold"].sum()
    )

    # -----------------------------
    # PRODUCT SELECTION
    # -----------------------------
    selected_product = st.selectbox(
        "Select Product",
        sorted(product_share.index.tolist())
    )

    product_units = (forecast_units * product_share[selected_product]).round(2)

    # -----------------------------
    # PRICE INPUT
    # -----------------------------
    price = st.number_input(
        f"Enter price per unit for {selected_product}",
        min_value=0.0,
        value=1.00,
        step=0.10
    )

    product_sales = (product_units * price).round(2)

    result = pd.DataFrame({
        "Month": product_units.index.strftime("%Y-%m"),
        "Predicted Units": product_units.values,
        "Expected Sales": product_sales.values
    })

    st.subheader("Product-Level Prediction Output")
    st.dataframe(result, use_container_width=True)

    # -----------------------------
    # SIMPLE VISUALS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(
            result,
            x="Month",
            y="Predicted Units",
            markers=True,
            title="Predicted Units (Product Level)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            result,
            x="Month",
            y="Expected Sales",
            title="Expected Sales (Based on Entered Price)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Units are forecasted using SARIMA at company level and distributed "
        "to products using historical contribution. Sales are calculated "
        "using user-defined product price."
    )

