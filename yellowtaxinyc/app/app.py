# app.py

import streamlit as st
import pandas as pd

from backend import load_artifacts, run_inference_on_dataframe

st.set_page_config(
    page_title="NYC Taxi Fare & Tip Prediction",
    layout="wide",
)

st.title("NYC Taxi Fare & Tip Prediction")

st.markdown(
    """
Upload a **NYC Yellow Taxi Parquet file** from the TLC trip record site.  
The app will use the **already-trained models** in `deployment/` to predict:
- expected **fare amount**
- **tip / no-tip** probability
"""
)


@st.cache_resource
def get_artifacts():
    scaler, fare_model, tip_model, feature_columns = load_artifacts()
    return scaler, fare_model, tip_model, feature_columns


uploaded_file = st.file_uploader(
    "Upload a TLC Parquet file (e.g., `yellow_tripdata_2023-01.parquet`)",
    type=["parquet"],
)

if uploaded_file is not None:
    df_raw = pd.read_parquet(uploaded_file)

    scaler, fare_model, tip_model, feature_columns = get_artifacts()

    with st.spinner("Running predictions with existing models..."):
        df_pred, metrics = run_inference_on_dataframe(
            df_raw, scaler, fare_model, tip_model, feature_columns
        )

    if df_pred.empty:
        st.error("No valid rows after cleaning. Check the uploaded file.")
    else:
        st.success(f"Predictions complete. Rows after cleaning: {len(df_pred):,}")

        # Metrics (if available)
        if metrics:
            st.subheader("Model Metrics (on uploaded data)")
            cols = st.columns(len(metrics))
            for (name, value), c in zip(metrics.items(), cols):
                c.metric(name, f"{value:.3f}")

        st.markdown("---")

        max_display = min(1000, len(df_pred))
        n_rows = st.slider(
            "Number of rows to display",
            min_value=10,
            max_value=max_display,
            value=min(100, max_display),
            step=10,
        )

        df_left = df_pred.head(n_rows)[
            [c for c in df_pred.columns if c in
             ["tpep_pickup_datetime", "trip_distance", "fare_amount", "tip_amount",
              "passenger_count", "payment_type"]]
        ]
        df_right = df_pred.head(n_rows)[
            ["pred_fare", "pred_tip_bool", "pred_tip_prob"]
        ]

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Raw Trips")
            st.dataframe(df_left, use_container_width=True)

        with col_right:
            st.subheader("Model Predictions")
            st.dataframe(df_right, use_container_width=True)
else:
    st.info(
        "Download a Yellow Taxi Parquet file from "
        "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page, "
        "then upload it here."
    )
