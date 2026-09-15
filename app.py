"""
app.py
------
Streamlit web application for King County House Price Prediction.

Run locally:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---- Paths -------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "metadata.json")

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Load model + metadata ---------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


model, meta = load_model()

# ---- Header ------------------------------------------------------------------
st.title("🏠 King County House Price Predictor")
st.markdown(
    "Enter the property details in the sidebar to get an **instant price estimate** "
    "powered by a machine-learning model trained on real King County (WA) sales data."
)

if model is None:
    st.error(
        "Model not found. Please run `python train_model.py` first to train and save the model."
    )
    st.stop()

# ---- Model info banner -------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Best Model",  meta["model_name"])
col2.metric("Test R²",     f"{meta['test_r2']:.4f}")
col3.metric("Test MAE",    f"${meta['test_mae']:,.0f}")

st.divider()

# ---- Sidebar inputs ----------------------------------------------------------
st.sidebar.header("Property Details")

fr = meta["feature_ranges"]  # convenient alias


def _int_range(key):
    return int(fr[key]["min"]), int(fr[key]["max"]), int(fr[key]["mean"])


bedrooms     = st.sidebar.number_input("Bedrooms",    min_value=0,   max_value=9,       value=3,    step=1)
bathrooms    = st.sidebar.number_input("Bathrooms",   min_value=0.0, max_value=8.0,     value=2.0,  step=0.25)
sqft_living  = st.sidebar.number_input("Living Area (sqft)",   min_value=200,    max_value=14000,   value=2000,  step=50)
sqft_lot     = st.sidebar.number_input("Lot Size (sqft)",      min_value=500,    max_value=1700000, value=7500,  step=100)
floors       = st.sidebar.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0], index=0)
waterfront   = st.sidebar.selectbox("Waterfront", options=["No", "Yes"], index=0)
waterfront_val = 1 if waterfront == "Yes" else 0
view         = st.sidebar.slider("View Quality (0=None, 4=Excellent)", min_value=0, max_value=4, value=0)
condition    = st.sidebar.slider("Condition (1=Poor, 5=Excellent)",    min_value=1, max_value=5, value=3)
sqft_above   = st.sidebar.number_input("Sqft Above Ground",    min_value=200,  max_value=10000, value=1500, step=50)
sqft_basement= st.sidebar.number_input("Sqft Basement",        min_value=0,    max_value=5000,  value=0,    step=50)
yr_built     = st.sidebar.number_input("Year Built",           min_value=1900, max_value=2015,  value=1990, step=1)
yr_renovated = st.sidebar.number_input(
    "Year Renovated (0 = never)", min_value=0, max_value=2015, value=0, step=1
)
sale_year    = st.sidebar.selectbox("Sale Year",   options=[2014, 2015], index=0)
sale_month   = st.sidebar.slider("Sale Month",     min_value=1, max_value=12, value=5)
city         = st.sidebar.selectbox(
    "City",
    options=meta["cities"],
    index=meta["cities"].index("Seattle") if "Seattle" in meta["cities"] else 0,
)
statezip     = st.sidebar.selectbox("State/ZIP", options=meta["statezips"], index=0)

# ---- Predict button ----------------------------------------------------------
predict_btn = st.sidebar.button("Predict Price", use_container_width=True, type="primary")

if predict_btn:
    input_df = pd.DataFrame([{
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms,
        "sqft_living":   sqft_living,
        "sqft_lot":      sqft_lot,
        "floors":        floors,
        "waterfront":    waterfront_val,
        "view":          view,
        "condition":     condition,
        "sqft_above":    sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built":      yr_built,
        "yr_renovated":  yr_renovated,
        "sale_year":     sale_year,
        "sale_month":    sale_month,
        "city":          city,
        "statezip":      statezip,
    }])

    raw_pred = model.predict(input_df)[0]

    # Undo log1p transform if model was trained on log-price
    if meta.get("log_transform_target", False):
        prediction = float(np.expm1(raw_pred))
    else:
        prediction = float(raw_pred)

    prediction = max(prediction, 0)

    st.success("### Estimated House Price")
    st.metric(
        label="Predicted Sale Price",
        value=f"${prediction:,.0f}",
        help=f"Based on {meta['model_name']} (Test R² = {meta['test_r2']:.4f})",
    )

    # ---- Input summary -------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Input Summary")
    summary_df = pd.DataFrame({
        "Feature": [
            "Bedrooms", "Bathrooms", "Living Area (sqft)", "Lot Size (sqft)",
            "Floors", "Waterfront", "View", "Condition",
            "Sqft Above Ground", "Sqft Basement", "Year Built", "Year Renovated",
            "Sale Year", "Sale Month", "City", "State/ZIP",
        ],
        "Value": [
            bedrooms, bathrooms, f"{sqft_living:,}", f"{sqft_lot:,}",
            floors, waterfront, view, condition,
            f"{sqft_above:,}", sqft_basement,
            yr_built, yr_renovated if yr_renovated > 0 else "Never",
            sale_year, sale_month, city, statezip,
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    st.info("Fill in the property details in the sidebar and click **Predict Price**.")

# ---- Model Comparison & Dataset Explorer ------------------------------------
st.divider()

tab1, tab2 = st.tabs(["Model Comparison", "Dataset Preview"])

with tab1:
    if "all_model_results" in meta:
        rows = []
        for model_name, mres in meta["all_model_results"].items():
            rows.append({
                "Model":       model_name,
                "CV R²":       mres["cv_r2_mean"],
                "Test R²":     mres["test_r2"],
                "Test MAE":    f"${mres['test_mae']:,.0f}",
                "Test RMSE":   f"${mres['test_rmse']:,.0f}",
                "Selected":    "YES" if model_name == meta["model_name"] else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        f"**Selected model:** `{meta['model_name']}` — highest Test R² score.  \n"
        "All models trained on `log1p(price)` to reduce skew; predictions are back-transformed with `expm1`."
    )

with tab2:
    data_path = os.path.join(BASE_DIR, "data", "modified_data.csv")
    if os.path.exists(data_path):
        df_preview = pd.read_csv(data_path)
        st.markdown(f"**Dataset:** `data/modified_data.csv` — "
                    f"{df_preview.shape[0]:,} rows x {df_preview.shape[1]} columns")
        st.markdown("**Target:** `price` (continuous, regression)")
        st.dataframe(df_preview.head(10), use_container_width=True)
    else:
        st.warning("Dataset file not found.")

# ---- Footer ------------------------------------------------------------------
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px;margin-top:40px;"
    "border-top:1px solid #e0e0e0;padding-top:10px;'>"
    "King County House Price Predictor · Streamlit + scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
