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
    page_title="Pharma Sales Analytics & Forecasting",
    layout="wide"
)

st.title("Pharma Sales Analytics & Forecasting Dashboard")
st.caption("Unit-based demand forecasting with price-driven sales scenarios")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# DATA FILES
# ==================================================
FILE_MAP = {
    "monthly_sales": "cleaned_monthly_sales.csv",
    "monthly_product_sales": "cleaned_monthly_product_sales.csv",
    "price_sensitivity": "cleaned_price_sensitivity.csv"
}

@st.cache_data
def load_data():
    data = {}
    for key, file in FILE_MAP.items():
        path = os.path.join(BASE_DIR, file)
        if not os.path.exists(path):
            st.error(f"Missing file: {file}")
            st.stop()

        df = pd.read_csv(path)
        data[key] = df

    return data

data = load_data()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Executive Overview",
        "Forecast Units & Sales",
        "Product Demand Planning"
    ]
)

# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================
if page == "Executive Overview":

    df = data["monthly_sales"].copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")

    latest = df.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Month Units", f"{int(latest['TotalUnits']):,}")
    c2.metric("Latest Month Sales", f"{int(latest['TotalSales']):,}")
    c3.metric("Avg Monthly Units", f"{int(df['TotalUnits'].mean()):,}")

    st.subheader("Historical Units Trend")

    fig = px.line(
        df,
        x="MonthStart",
        y="TotalUnits",
        markers=True,
        labels={
            "MonthStart": "Month",
            "TotalUnits": "Units Sold"
        },
        title="Monthly Units Sold (Actual)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# FORECAST UNITS & SALES
# ==================================================
elif page == "Forecast Units & Sales":

    st.subheader("Unit Demand Forecast (Inflation-Free)")

    df = data["monthly_sales"].copy()
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

    horizon = st.slider("Forecast months", 3, 6, 6)

    forecast = res.get_forecast(steps=horizon).summary_frame()
    forecast_units = forecast["mean"].round(0).astype(int)

    # ==============================
    # SIMPLE FORECAST CHART
    # ==============================
    st.subheader("Actual vs Predicted Units")

    hist_df = ts.reset_index()
    hist_df.columns = ["Date", "Units"]

    forecast_df = forecast_units.reset_index()
    forecast_df.columns = ["Date", "Predicted Units"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_df["Date"],
        y=hist_df["Units"],
        name="Actual Units",
        mode="lines+markers"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Predicted Units"],
        name="Predicted Units",
        mode="lines+markers"
    ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Units",
        title="Units Forecast (Actual vs Predicted)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # PRICE INPUT → SALES ESTIMATION
    # ==============================
    st.subheader("Price-Based Sales Estimation")

    avg_price = data["price_sensitivity"]["AvgSellingPrice"].mean()
    entered_price = st.number_input(
        "Enter expected average price per unit",
        min_value=0.0,
        value=float(round(avg_price, 2))
    )

    sales_estimate = forecast_units * entered_price

    result = pd.DataFrame({
        "Predicted Units": forecast_units.values,
        "Estimated Sales": sales_estimate.round(0).astype(int).values
    }, index=forecast_units.index)

    st.dataframe(result, use_container_width=True)

# ==================================================
# PRODUCT DEMAND PLANNING
# ==================================================
elif page == "Product Demand Planning":

    st.subheader("Products Expected to Sell More")

    prod = data["monthly_product_sales"].copy()
    prod["UnitsSold"] = pd.to_numeric(prod["UnitsSold"], errors="coerce").fillna(0)

    product_share = (
        prod.groupby("ProductName")["UnitsSold"].sum()
        / prod["UnitsSold"].sum()
    ).sort_values(ascending=False)

    # Use latest forecasted units
    df = data["monthly_sales"].copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    ts = df.set_index("MonthStart")["TotalUnits"].asfreq("MS").fillna(0)

    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    future_units = int(res.get_forecast(steps=1).summary_frame()["mean"].iloc[0])

    product_forecast = (
        product_share * future_units
    ).round(0).astype(int).reset_index()

    product_forecast.columns = ["Product", "Expected Units"]

    st.dataframe(product_forecast.head(10), use_container_width=True)

    fig = px.bar(
        product_forecast.head(10),
        x="Expected Units",
        y="Product",
        orientation="h",
        labels={
            "Expected Units": "Predicted Units",
            "Product": "Product Name"
        },
        title="Top Products by Expected Demand"
    )

    st.plotly_chart(fig, use_container_width=True)

