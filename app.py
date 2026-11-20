from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import numpy as np

csv_path = Path(__file__).with_name("customer_segments.csv")
customers = pd.read_csv(csv_path)

st.title("Customer Segmentation & Marketing Intelligence")

left, middle, right = st.columns(3)
left.metric("Customers", f"{customers['CustomerID'].nunique():,}")
middle.metric("Segments", customers["Segment"].nunique())
right.metric("Recorded purchase value", f"£{customers['Monetary'].sum():,.2f}")

st.subheader("Customers by segment")
segment_counts = customers["Segment"].value_counts()
st.bar_chart(segment_counts.rename("Customers"))
st.subheader("Segment profiles and campaign ideas")

campaign_ideas = {
    "Active High-Value Buyers": "Offer loyalty rewards or early access to new products.",
    "Occasional Buyers": "Recommend related products to encourage another order.",
    "Inactive Low-Frequency Buyers": "Test a win-back message with a small incentive."
}

profiles = customers.groupby("Segment").agg(
    Customers=("CustomerID", "size"),
    Median_Recency=("Recency", "median"),
    Median_Orders=("Frequency", "median"),
    Median_Purchase_Value=("Monetary", "median")
)

profiles["Campaign to test"] = profiles.index.map(campaign_ideas)

st.dataframe(profiles.round(2).rename(columns={
    "Median_Recency": "Median days since last purchase",
    "Median_Orders": "Median orders",
    "Median_Purchase_Value": "Median purchase value (£)"
}))

st.subheader("Look up an existing customer")

customer_id = st.selectbox(
    "Customer ID",
    options=sorted(customers["CustomerID"].unique()),
    index=None,
    placeholder="Type or select a customer ID"
)

if customer_id is not None:
    customer = customers.loc[customers["CustomerID"] == customer_id].iloc[0]

    st.write("**Segment:**", customer["Segment"])

    left, middle, right = st.columns(3)
    left.metric("Days since last purchase", int(customer["Recency"]))
    middle.metric("Orders", int(customer["Frequency"]))
    right.metric("Recorded purchase value", f"£{customer['Monetary']:,.2f}")
    st.subheader("Assign a segment from purchase history")

with st.form("segment_form"):
    recency = st.number_input("Days since last purchase", min_value=0, value=20)
    frequency = st.number_input("Number of orders", min_value=1, value=6)
    monetary = st.number_input("Recorded purchase value (£)", min_value=0.01, value=2000.0)
    submitted = st.form_submit_button("Assign segment")

if submitted:
    model_path = Path(__file__).with_name("customer_segmentation_model.joblib")
    bundle = joblib.load(model_path)

    customer_rfm = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=bundle["rfm_columns"]
    )

    log_columns = bundle["log_columns"]
    customer_rfm[log_columns] = np.log1p(customer_rfm[log_columns])

    scaled_rfm = bundle["scaler"].transform(customer_rfm)
    cluster = int(bundle["kmeans"].predict(scaled_rfm)[0])

    st.success(f"Assigned segment: {bundle['segment_names'][cluster]}")

    st.subheader("Customer share vs. purchase value")

segment_summary = customers.groupby("Segment").agg(
    Customers=("CustomerID", "size"),
    Purchase_Value=("Monetary", "sum")
)

segment_shares = pd.DataFrame({
    "Customer share (%)": (
        100 * segment_summary["Customers"] / segment_summary["Customers"].sum()
    ),
    "Purchase value share (%)": (
        100 * segment_summary["Purchase_Value"]
        / segment_summary["Purchase_Value"].sum()
    )
}).sort_values("Purchase value share (%)", ascending=False)

st.bar_chart(segment_shares)
st.dataframe(segment_shares.round(2))