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
    page_title="Product Sales Forecasting Dashboard",
    layout="wide"
)

st.title("Product-Level Sales Forecasting Dashboard")
st.caption("Unit-based forecasting with product-specific price scenarios")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    monthly_sales = pd.read_csv("cleaned_monthly_sales.csv")
    product_sales = pd.read_csv("cleaned_monthly_product_sales.csv")
    return monthly_sales, product_sales

monthly_sales, product_sales = load_data()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Product Sales Forecast"
    ]
)

# ==================================================
# PRODUCT SALES FORECAST
# ==================================================
if page == "Product Sales Forecast":

    st.subheader("Product-Wise Sales Forecast (Price Adjustable)")

    # -----------------------------
    # PREPARE COMPANY UNITS SERIES
    # -----------------------------
    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")
    df["TotalUnits"] = pd.to_numeric(df["TotalUnits"], errors="coerce").fillna(0)

    ts = df.set_index("MonthStart")["TotalUnits"].asfreq("MS").fillna(0)

    # -----------------------------
    # SARIMA MODEL
    # -----------------------------
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
    forecast_units = forecast["mean"]

    # -----------------------------
    # PRODUCT SHARE CALCULATION
    # -----------------------------
    product_sales["UnitsSold"] = pd.to_numeric(
        product_sales["UnitsSold"], errors="coerce"
    ).fillna(0)

    product_share = (
        product_sales.groupby("ProductName")["UnitsSold"].sum()
        / product_sales["UnitsSold"].sum()
    )

    # -----------------------------
    # PRODUCT SELECTION
    # -----------------------------
    selected_product = st.selectbox(
        "Select Product",
        sorted(product_share.index.tolist())
    )

    # Product-specific units forecast
    product_units = forecast_units * product_share[selected_product]

    # -----------------------------
    # PRICE INPUT
    # -----------------------------
    price = st.number_input(
        f"Enter price per unit for {selected_product}",
        min_value=0.0,
        value=1.00,
        step=0.10
    )

    # -----------------------------
    # SALES CALCULATION
    # -----------------------------
    sales = product_units * price

    result = pd.DataFrame({
        "Month": product_units.index.strftime("%Y-%m"),
        "Predicted Units": product_units.round(2),
        "Expected Sales": sales.round(2)
    })

    st.subheader("Predicted Units & Expected Sales (Product-Specific)")
    st.dataframe(result, use_container_width=True)

    # -----------------------------
    # SIMPLE VISUALS (EASY TO EXPLAIN)
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig_units = px.line(
            result,
            x="Month",
            y="Predicted Units",
            markers=True,
            title="Predicted Units (Product Level)"
        )
        st.plotly_chart(fig_units, use_container_width=True)

    with col2:
        fig_sales = px.bar(
            result,
            x="Month",
            y="Expected Sales",
            title="Expected Sales (Based on Entered Price)"
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    st.caption(
        "Note: Units are forecasted using historical seasonality. "
        "Sales are calculated by multiplying forecasted units with user-entered product price."
    )
