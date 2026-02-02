# SuperKart Sales Forecasting - Streamlit Frontend
# Interactive UI for sales prediction

import streamlit as st
import requests

st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Forecaster")
st.markdown("*ML-powered sales prediction for retail inventory optimization*")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Product Information")
    
    Product_Weight = st.number_input(
        "Product Weight (kg)", 
        min_value=4.0, 
        max_value=22.0,
        value=12.66,
        step=0.1
    )
    
    Product_Sugar_Content = st.selectbox(
        "Sugar Content", 
        ["Low Sugar", "Regular", "No Sugar"]
    )
    
    Product_Allocated_Area = st.slider(
        "Display Area Ratio", 
        min_value=0.0, 
        max_value=0.3, 
        value=0.07,
        step=0.01
    )
    
    Product_MRP = st.number_input(
        "Product MRP ($)", 
        min_value=31.0, 
        max_value=270.0,
        value=150.0,
        step=1.0
    )
    
    Product_Id_char = st.selectbox(
        "Product Category",
        ["FD", "DR", "NC"],
        help="FD=Food, DR=Drinks, NC=Non-Consumables"
    )
    
    Product_Type_Category = st.selectbox(
        "Product Type",
        ["Perishables", "Non Perishables"]
    )

with col2:
    st.subheader("🏪 Store Information")
    
    Store_Size = st.selectbox(
        "Store Size",
        ["Small", "Medium", "High"]
    )
    
    Store_Location_City_Type = st.selectbox(
        "City Tier",
        ["Tier 1", "Tier 2", "Tier 3"],
        help="Tier 1 = Metro, Tier 2 = Mid-size, Tier 3 = Small towns"
    )
    
    Store_Type = st.selectbox(
        "Store Type",
        ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
    )
    
    Store_Age_Years = st.slider(
        "Store Age (Years)",
        min_value=1,
        max_value=40,
        value=15
    )

st.markdown("---")

# Prepare payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Backend API URL
API_URL = "https://rnunez245-superkart-backend.hf.space/v1/predict"

if st.button("🔮 Predict Sales", type='primary', use_container_width=True):
    with st.spinner("Calculating prediction..."):
        try:
            response = requests.post(API_URL, json=product_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                predicted_sales = result["Sales"]
                
                st.success("Prediction Complete!")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Quarterly Sales", f"${predicted_sales:,.2f}")
                with col_b:
                    st.metric("Monthly Estimate", f"${predicted_sales/3:,.2f}")
                with col_c:
                    st.metric("Annual Projection", f"${predicted_sales*4:,.2f}")
            else:
                st.error(f"API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>Built by Ruben Nunez | UT Austin AI/ML Program</p>
    <p>Model: Random Forest Regressor (R² = 0.63)</p>
</div>
""", unsafe_allow_html=True)
