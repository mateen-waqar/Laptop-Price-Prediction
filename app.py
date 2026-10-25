"""
Laptop Price Prediction - Streamlit web app
AIC354 Machine Learning Fundamentals - Assignment 3
S.M.MATEEN UD DIN (FA24-BSE-152)

Loads the trained pipeline (laptop_price_model.pkl) and predicts the price (PKR)
of a laptop from its specifications.
"""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")


@st.cache_resource
def load_model():
    with open("laptop_price_model.pkl", "rb") as f:
        return pickle.load(f)


package = load_model()
model = package["model"]
feature_names = package["feature_names"]
options = package["options"]

# The model was trained on prices in euros (dataset column Price_euros).
# All values are converted to Pakistani rupees for display with this rate.
# Update it if you want a newer exchange rate (about 316 PKR per EUR in mid-2026).
EUR_TO_PKR = 316.0

rmse = package["cv_rmse_mean"] * EUR_TO_PKR
rmse_std = package["cv_rmse_std"] * EUR_TO_PKR
baseline_rmse = package["baseline_rmse"] * EUR_TO_PKR

RESOLUTIONS = ["1366x768", "1600x900", "1920x1080", "2560x1440", "2560x1600",
               "2880x1800", "3200x1800", "3840x2160"]

st.title("💻 Laptop Price Prediction")
st.write(
    "Estimates the market price of a laptop from its specifications, using a "
    f"**{package['algorithm']}** model selected by 10-fold cross validation "
    "on 1,303 laptops (prices shown in Pakistani rupees, PKR)."
)

c1, c2, c3 = st.columns(3)
c1.metric("Model", package["algorithm"])
c2.metric("CV RMSE", f"{rmse:,.0f} PKR", f"± {rmse_std:,.0f}", delta_color="off")
c3.metric("Improvement over average-price guess",
          f"{(1 - rmse / baseline_rmse) * 100:.0f}%")

st.divider()

left, mid, right = st.columns(3)

with left:
    st.subheader("1. General")
    company = st.selectbox("Brand", options["Company"], index=options["Company"].index("Dell"))
    type_name = st.selectbox("Laptop type", options["TypeName"], index=options["TypeName"].index("Notebook"))
    os_group = st.selectbox("Operating system", options["OS_Group"], index=options["OS_Group"].index("Windows"))
    weight = st.slider("Weight (kg)", 0.7, 4.7, 2.0, 0.1)

with mid:
    st.subheader("2. Screen")
    inches = st.slider("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    resolution = st.selectbox("Resolution", RESOLUTIONS, index=2)
    ips = st.checkbox("IPS panel", value=True)
    touchscreen = st.checkbox("Touchscreen", value=False)

with right:
    st.subheader("3. Performance")
    cpu_family = st.selectbox("Processor family", options["Cpu_Family"],
                              index=options["Cpu_Family"].index("Intel Core i5"))
    cpu_ghz = st.slider("Processor speed (GHz)", 0.9, 3.6, 2.5, 0.1)
    gpu_brand = st.selectbox("Graphics card maker", options["Gpu_Brand"],
                             index=options["Gpu_Brand"].index("Intel"))
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)

st.subheader("4. Storage")
s1, s2, s3 = st.columns(3)
ssd = s1.selectbox("SSD (GB)", [0, 128, 256, 512, 1024], index=2)
hdd = s2.selectbox("HDD (GB)", [0, 500, 1000, 2000], index=0)
other = s3.selectbox("Flash / hybrid storage (GB)", [0, 32, 64, 128, 256, 512], index=0)

width, height = (int(v) for v in resolution.split("x"))
features = {
    "Inches": inches,
    "Ram_GB": ram,
    "Weight_kg": weight,
    "ppi": float(np.sqrt(width ** 2 + height ** 2) / inches),
    "Cpu_GHz": cpu_ghz,
    "SSD_GB": ssd,
    "HDD_GB": hdd,
    "Other_Storage_GB": other,
    "Touchscreen": int(touchscreen),
    "IPS": int(ips),
    "Company": company,
    "TypeName": type_name,
    "Cpu_Family": cpu_family,
    "Gpu_Brand": gpu_brand,
    "OS_Group": os_group,
}
X = pd.DataFrame([features])[feature_names]

st.divider()
if st.button("🔮 Predict price", type="primary", width="stretch"):
    price = float(model.predict(X)[0]) * EUR_TO_PKR
    st.success(f"### Estimated price: **Rs {price:,.0f} PKR**")
    st.write(f"Typical error of this model: about **± Rs {rmse:,.0f}**, so the likely range is "
             f"**Rs {max(price - rmse, 0):,.0f} – Rs {price + rmse:,.0f}**.")
    if ssd == 0 and hdd == 0 and other == 0:
        st.warning("No storage selected - please choose at least one drive for a realistic estimate.")

with st.expander("Model comparison (10-fold cross validation)"):
    cv = pd.DataFrame(package["cv_summary"]).set_index("Algorithm")
    cv = cv.rename(columns={"Mean RMSE (EUR)": "Mean RMSE (PKR)", "Std RMSE (EUR)": "Std RMSE (PKR)"})
    for col in ["Mean RMSE (PKR)", "Std RMSE (PKR)"]:
        if col in cv.columns:
            cv[col] = (cv[col] * EUR_TO_PKR).round(0)
    st.dataframe(cv, width="stretch")
    st.caption(f"Baseline (always the average price) RMSE: Rs {baseline_rmse:,.0f} PKR  "
               f"(converted at 1 EUR = {EUR_TO_PKR:.0f} PKR)")

st.caption("AIC354 Machine Learning Fundamentals – Assignment 3 · S.M.MATEEN UD DIN (FA24-BSE-152) · "
           "COMSATS University Islamabad, Lahore Campus")
