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
st.set_page_config(page_title="Pharma Analytics & Forecasting", layout="wide")

st.title("Pharma Sales Analytics & Forecasting Dashboard")
st.caption(
    "Analytics, unit-based demand forecasting, and product-wise sales planning"
)

# ==================================================
# CACHE CONTROL (🔥 FIX)
# ==================================================
if st.sidebar.button("🔄 Refresh Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()

def file_signature(path):
    return os.path.getmtime(path)

sig_monthly_sales = file_signature("cleaned_monthly_sales.csv")
sig_monthly_product_sales = file_signature("cleaned_monthly_product_sales.csv")

# ==================================================
# LOAD DATA (CACHE-SAFE)
# ==================================================
@st.cache_data
def load_data(sig1, sig2):
    ms = pd.read_csv("cleaned_monthly_sales.csv")
    mps = pd.read_csv("cleaned_monthly_product_sales.csv")
    return ms, mps

monthly_sales, monthly_product_sales = load_data(
    sig_monthly_sales, sig_monthly_product_sales
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Executive Analysis", "Units Forecast (Company)", "Product Forecast & Sales Planning"]
)

# ==================================================
# EXECUTIVE ANALYSIS
# ==================================================
if page == "Executive Analysis":

    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")

    latest = df.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Units", round(latest["TotalUnits"], 2))
    c2.metric("Latest Sales", round(latest["TotalSales"], 2))
    c3.metric("Avg Monthly Units", round(df["TotalUnits"].mean(), 2))

    fig = px.line(
        df,
        x="MonthStart",
        y="TotalUnits",
        markers=True,
        title="Monthly Units Sold (Historical)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# COMPANY LEVEL FORECAST
# ==================================================
elif page == "Units Forecast (Company)":

    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")

    ts = df.set_index("MonthStart")["TotalUnits"]

    full_index = pd.date_range(
        start=ts.index.min(),
        end=ts.index.max(),
        freq="MS"
    )
    ts = ts.reindex(full_index)
    ts = ts.interpolate(method="linear")

    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    horizon = st.slider("Forecast Months", 3, 6, 6)

    forecast_units = (
        res.get_forecast(steps=horizon)
        .summary_frame()["mean"]
        .abs()
        .clip(lower=0)
        .round(2)
    )

    hist_df = ts.reset_index()
    hist_df.columns = ["Month", "Actual Units"]

    fc_df = forecast_units.reset_index()
    fc_df.columns = ["Month", "Predicted Units"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Month"], y=hist_df["Actual Units"], name="Actual"))
    fig.add_trace(go.Scatter(x=fc_df["Month"], y=fc_df["Predicted Units"], name="Forecast"))

    fig.update_layout(
        title="Company-Level Units Forecast",
        xaxis_title="Month",
        yaxis_title="Units"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fc_df, use_container_width=True)

# ==================================================
# PRODUCT FORECAST & SALES
# ==================================================
elif page == "Product Forecast & Sales Planning":

    df = monthly_sales.copy()
    df["MonthStart"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    )
    df = df.sort_values("MonthStart")

    ts = df.set_index("MonthStart")["TotalUnits"]

    full_index = pd.date_range(ts.index.min(), ts.index.max(), freq="MS")
    ts = ts.reindex(full_index).interpolate(method="linear")

    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    horizon = st.slider("Forecast Months ", 3, 6, 6)
    forecast_units = (
        res.get_forecast(steps=horizon)
        .summary_frame()["mean"]
        .abs()
        .clip(lower=0)
    )

    monthly_product_sales["UnitsSold"] = pd.to_numeric(
        monthly_product_sales["UnitsSold"], errors="coerce"
    ).fillna(0)

    product_totals = monthly_product_sales.groupby("ProductName")["UnitsSold"].sum()
    product_share = (product_totals / product_totals.sum()).fillna(0).clip(lower=0)

    selected_product = st.selectbox("Select Product", sorted(product_share.index))

    product_units = (forecast_units * product_share[selected_product]).round(2)
    product_units = np.maximum(product_units, 0)

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

    st.dataframe(result, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.line(
                result,
                x="Month",
                y="Predicted Units",
                markers=True,
                title="Predicted Units (Selected Product)"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.bar(
                result,
                x="Month",
                y="Expected Sales",
                title="Expected Sales (Based on Price)"
            ),
            use_container_width=True
        )

    st.caption(
        "Forecast starts from Feb 2026 (first unseen month). "
        "Units are constrained to be non-negative and distributed using historical product contribution."
    )

