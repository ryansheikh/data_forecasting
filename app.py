import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Pharmevo Sales Analytics", layout="wide")
st.title("Pharmevo Sales Analytics Dashboard")
st.caption("Executive analytics and unit forecasting built on aggregated SQL Server exports")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_MAP = {
    "monthly_sales": "cleaned_monthly_sales.csv",
    "bonus_discount_monthly": "cleaned_bonus_discount_monthly.csv",
    "top_products": "cleaned_top_products.csv",
    "monthly_product_sales": "cleaned_monthly_product_sales.csv",
    "distributor_performance": "cleaned_distributor_performance.csv",
    "client_type_analysis": "cleaned_client_type_analysis.csv",
    "monthly_client_type_sales": "cleaned_monthly_client_type_sales.csv",
    "seasonality_monthly_avg": "cleaned_seasonality_monthly_avg.csv",
    "price_sensitivity": "cleaned_price_sensitivity.csv",
    "dimension_summary": "cleaned_dimension_summary.csv",
}

NUMERIC_COLUMNS = {
    "TotalUnits", "TotalBonus", "TotalDiscount", "TotalSales",
    "UnitsSold", "Revenue", "TotalClients",
    "AvgSellingPrice", "AvgMonthlySales"
}

@st.cache_data
def load_data():
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
                df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
                errors="coerce"
            )
        data[key] = df

    return data

def fmt(x):
    x = float(x)
    if abs(x) >= 1e9: return f"{x/1e9:.2f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.2f}K"
    return f"{x:,.0f}"

data = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Executive Overview",
        "Forecast Units",
        "Product Performance",
        "Distributor Performance",
        "Client Analysis",
        "Promotion Impact",
        "Seasonality & Cycles",
        "Pricing Analysis",
        "Dimension Drilldown",
    ]
)

# -------------------------
# Executive Overview
# -------------------------
if page == "Executive Overview":
    df = data["monthly_sales"].sort_values("MonthStart").copy()
    df["MoM_Growth"] = df["TotalSales"].pct_change() * 100
    df["Rolling_3M"] = df["TotalSales"].rolling(3).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Sales", fmt(latest["TotalSales"]),
              f"{((latest['TotalSales']-prev['TotalSales'])/prev['TotalSales']*100):.2f}%" if prev["TotalSales"] else None)
    k2.metric("Latest Units", fmt(latest["TotalUnits"]))
    k3.metric("Avg Monthly Sales", fmt(df["TotalSales"].mean()))
    k4.metric("Avg Monthly Units", fmt(df["TotalUnits"].mean()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["TotalSales"], name="Sales"))
    fig.add_trace(go.Scatter(x=df["MonthStart"], y=df["Rolling_3M"], name="3M Rolling Avg"))
    fig.update_layout(title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(df, x="MonthStart", y="MoM_Growth", title="Month-on-Month Sales Growth (%)")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Forecast Units (ML)
# -------------------------
elif page == "Forecast Units":
    df = data["monthly_sales"].sort_values("MonthStart").copy()
    df = df.dropna(subset=["MonthStart"])
    df["TotalUnits"] = pd.to_numeric(df["TotalUnits"], errors="coerce").fillna(0)

    ts = df.set_index("MonthStart")[["TotalUnits"]].asfreq("MS")
    ts["TotalUnits"] = ts["TotalUnits"].fillna(0)

    st.subheader("Train/Test Split")
    st.write("Train: 2024-01 to 2025-12")
    st.write("Test: 2026-01")

    train = ts.loc["2024-01-01":"2025-12-01", "TotalUnits"]
    test = ts.loc["2026-01-01":"2026-01-01", "TotalUnits"]

    # Small tuning grid (fast and explainable)
    candidate_models = [
        ((1,1,1), (1,1,1,12)),
        ((2,1,1), (1,1,1,12)),
        ((1,1,2), (1,1,1,12)),
        ((2,1,2), (1,1,1,12)),
        ((1,1,1), (1,1,0,12)),
        ((1,1,1), (0,1,1,12)),
    ]

    best = {"aic": np.inf, "order": None, "seasonal": None, "res": None}
    for order, seasonal in candidate_models:
        try:
            m = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            r = m.fit(disp=False)
            if r.aic < best["aic"]:
                best.update({"aic": r.aic, "order": order, "seasonal": seasonal, "res": r})
        except:
            continue

    st.write("Selected model:")
    st.write({"order": best["order"], "seasonal_order": best["seasonal"], "AIC": float(best["aic"])})

    # Evaluate on test (2026-01)
    if len(test) == 1:
        pred_test = best["res"].get_forecast(steps=1).summary_frame()
        y_hat = float(pred_test["mean"].iloc[0])
        y_true = float(test.iloc[0])
        mae = abs(y_true - y_hat)
        mape = (mae / y_true * 100) if y_true != 0 else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Units (2026-01)", fmt(y_true))
        c2.metric("Predicted Units (2026-01)", fmt(y_hat))
        c3.metric("MAPE (%)", f"{mape:.2f}" if not np.isnan(mape) else "NA")

    st.subheader("Forecast Horizon")
    horizon = st.slider("Months to forecast", min_value=3, max_value=6, value=6)

    # Fit final model on full series up to 2026-01
    full = ts["TotalUnits"].copy()
    final_model = SARIMAX(
        full,
        order=best["order"],
        seasonal_order=best["seasonal"],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    final_res = final_model.fit(disp=False)

    fc = final_res.get_forecast(steps=horizon).summary_frame()

    # Robust CI column detection
    lower_col = "mean_ci_lower" if "mean_ci_lower" in fc.columns else [c for c in fc.columns if "lower" in c][0]
    upper_col = "mean_ci_upper" if "mean_ci_upper" in fc.columns else [c for c in fc.columns if "upper" in c][0]

    out = fc[["mean", lower_col, upper_col]].copy()
    out.columns = ["Predicted_Units", "Lower_Bound", "Upper_Bound"]
    out = out.round(0).astype(int)

    st.subheader("Forecast Output (Units)")
    st.dataframe(out, use_container_width=True)

    # Chart: history + forecast
    hist = full.reset_index().rename(columns={"MonthStart":"Date", "TotalUnits":"Units"})
    fc_plot = out.reset_index().rename(columns={"index":"Date"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Units"], name="Actual Units"))
    fig.add_trace(go.Scatter(x=fc_plot["Date"], y=fc_plot["Predicted_Units"], name="Forecast Units"))
    fig.add_trace(go.Scatter(
        x=fc_plot["Date"],
        y=fc_plot["Upper_Bound"],
        name="Upper Bound",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fc_plot["Date"],
        y=fc_plot["Lower_Bound"],
        name="Lower Bound",
        line=dict(dash="dot")
    ))
    fig.update_layout(title="Units Forecast (with Confidence Interval)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Product Performance
# -------------------------
elif page == "Product Performance":
    top = data["top_products"].sort_values("Revenue", ascending=False).head(20)
    fig = px.bar(top, x="Revenue", y="ProductName", orientation="h", title="Top Products by Revenue")
    st.plotly_chart(fig, use_container_width=True)

    mps = data["monthly_product_sales"].dropna(subset=["MonthStart"])
    products = sorted(mps["ProductName"].unique())
    selected = st.multiselect("Select Products", products, default=products[:1])
    if selected:
        fig2 = px.line(
            mps[mps["ProductName"].isin(selected)],
            x="MonthStart", y="Revenue", color="ProductName", markers=True,
            title="Product Revenue Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Distributor Performance
# -------------------------
elif page == "Distributor Performance":
    dist = data["distributor_performance"].sort_values("Revenue", ascending=False).head(30)
    fig = px.bar(dist, x="Revenue", y="DistributorName", orientation="h", title="Top Distributors by Revenue")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Client Analysis
# -------------------------
elif page == "Client Analysis":
    ct = data["client_type_analysis"]
    fig = px.pie(ct, names="ClientType", values="Revenue", title="Revenue Share by Client Type")
    st.plotly_chart(fig, use_container_width=True)

    mct = data["monthly_client_type_sales"].dropna(subset=["MonthStart"])
    fig2 = px.line(mct, x="MonthStart", y="Revenue", color="ClientType", markers=True, title="Client Type Trend")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Promotion Impact (no OLS to avoid extra deps beyond statsmodels already used)
# -------------------------
elif page == "Promotion Impact":
    promo = data["bonus_discount_monthly"].dropna(subset=["MonthStart"])
    sales = data["monthly_sales"][["MonthStart", "TotalSales"]].dropna(subset=["MonthStart"])

    merged = promo.merge(sales, on="MonthStart", how="inner")
    merged["Promo_Total"] = merged["TotalBonus"] + merged["TotalDiscount"]

    c1, c2 = st.columns(2)
    c1.metric("Avg Monthly Promotion", fmt(merged["Promo_Total"].mean()))
    ratio = (merged["Promo_Total"].sum() / merged["TotalSales"].sum() * 100) if merged["TotalSales"].sum() else 0
    c2.metric("Promotion to Sales Ratio", f"{ratio:.2f}%")

    fig = px.line(merged, x="MonthStart", y=["TotalBonus", "TotalDiscount"], markers=True,
                  title="Bonus and Discount Trend")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(merged, x="Promo_Total", y="TotalSales", title="Promotion Spend vs Sales")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Seasonality & Cycles
# -------------------------
elif page == "Seasonality & Cycles":
    sea = data["seasonality_monthly_avg"]
    fig = px.bar(sea, x="Month", y="AvgMonthlySales", title="Average Monthly Sales by Month (Seasonality)")
    st.plotly_chart(fig, use_container_width=True)

    dfm = data["monthly_sales"].copy()
    heat = dfm.pivot_table(index="Year", columns="Month", values="TotalSales", aggfunc="sum", fill_value=0)
    fig2 = px.imshow(heat, aspect="auto", title="Sales Heatmap (Year x Month)")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Pricing Analysis
# -------------------------
elif page == "Pricing Analysis":
    price = data["price_sensitivity"].sort_values("AvgSellingPrice", ascending=False).head(30)
    fig = px.bar(price, x="AvgSellingPrice", y="ProductName", orientation="h", title="Avg Selling Price by Product")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(data["price_sensitivity"], x="AvgSellingPrice", y="TotalUnits",
                      hover_data=["ProductName"], title="Price vs Volume")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Dimension Drilldown
# -------------------------
elif page == "Dimension Drilldown":
    dim = data["dimension_summary"]
    c1, c2, c3 = st.columns(3)
    with c1:
        distributor = st.selectbox("Distributor", ["All"] + sorted(dim["DistributorName"].unique()))
    with c2:
        client_type = st.selectbox("Client Type", ["All"] + sorted(dim["ClientType"].unique()))
    with c3:
        team = st.selectbox("Team", ["All"] + sorted(dim["TeamName"].unique()))

    df = dim.copy()
    if distributor != "All":
        df = df[df["DistributorName"] == distributor]
    if client_type != "All":
        df = df[df["ClientType"] == client_type]
    if team != "All":
        df = df[df["TeamName"] == team]

    summary = df.groupby("BrickName", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(summary, x="Revenue", y="BrickName", orientation="h", title="Revenue by Brick")
    st.plotly_chart(fig, use_container_width=True)

