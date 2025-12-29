import streamlit as st
import pandas as pd
import pickle

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.set_page_config(
    page_title="Hyderabad House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Hyderabad House Rent Prediction")
st.caption("XGBoost Machine Learning Model")

st.markdown("---")

# ================= USER INPUT =================
bedrooms = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
area = st.number_input("Area (sqft)", min_value=300, max_value=5000, value=1200)

furnishing = st.selectbox(
    "Furnishing",
    encoders["Furnishing"].classes_
)

tennants = st.selectbox(
    "Tenants",
    encoders["Tennants"].classes_
)

locality = st.selectbox(
    "Locality",
    encoders["Locality"].classes_
)

# ================= PREDICTION =================
if st.button("Predict Rent"):
    input_df = pd.DataFrame(
        [[bedrooms, bathrooms, furnishing, tennants, area, locality]],
        columns=["Bedrooms", "Bathrooms", "Furnishing", "Tennants", "Area", "Locality"]
    )

    # Encode categorical features
    input_df["Furnishing"] = encoders["Furnishing"].transform(input_df["Furnishing"])
    input_df["Tennants"] = encoders["Tennants"].transform(input_df["Tennants"])
    input_df["Locality"] = encoders["Locality"].transform(input_df["Locality"])

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Monthly Rent: ₹ {prediction:,.0f}")

st.markdown("---")
st.caption("Built with Streamlit & XGBoost")
