"""
SuperKart Sales Forecasting App
================================
Predicts product sales revenue for SuperKart retail chain stores.

Model: Random Forest Regressor (R² = 0.63)
Author: Ruben Nunez | UT Austin AI/ML Program (Top 5, GPA 4.28)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🛒 SuperKart Sales Forecaster</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered sales prediction for retail inventory optimization</p>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model from file."""
    model_path = "model/superkart_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("⚠️ Model file not found. Please ensure the model is trained and saved.")
        return None

model = load_model()

# Sidebar for model info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=80)
    st.markdown("### About This Model")
    st.markdown("""
    **Algorithm:** Random Forest Regressor
    
    **Performance Metrics:**
    - R² Score: 0.63
    - RMSE: ~590
    - MAPE: 17.8%
    
    **Features Used:**
    - Product attributes (weight, price, category)
    - Store characteristics (size, location, type)
    
    ---
    
    **Business Context:**
    SuperKart uses this model to forecast quarterly sales revenue, enabling:
    - Optimized inventory management
    - Regional sales strategy planning
    - Data-driven procurement decisions
    """)
    
    st.markdown("---")
    st.markdown("**Built by:** Ruben Nunez")
    st.markdown("**Program:** UT Austin AI/ML")
    st.markdown("[View on GitHub](https://github.com/rnunez245)")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📦 Product Information")
    
    product_weight = st.slider(
        "Product Weight (kg)",
        min_value=4.0,
        max_value=22.0,
        value=12.5,
        step=0.1,
        help="Weight of the product in kilograms"
    )
    
    product_mrp = st.slider(
        "Product MRP (₹)",
        min_value=31.0,
        max_value=270.0,
        value=150.0,
        step=1.0,
        help="Maximum Retail Price of the product"
    )
    
    product_allocated_area = st.slider(
        "Display Area Ratio",
        min_value=0.0,
        max_value=0.3,
        value=0.07,
        step=0.01,
        help="Ratio of allocated display area to total store display area"
    )
    
    product_sugar_content = st.selectbox(
        "Sugar Content",
        options=["Low Sugar", "Regular", "No Sugar"],
        help="Sugar content level of the product"
    )
    
    product_id_char = st.selectbox(
        "Product Category Code",
        options=["FD", "DR", "NC"],
        help="FD=Food, DR=Drinks, NC=Non-Consumables"
    )
    
    product_type_category = st.selectbox(
        "Product Type",
        options=["Perishables", "Non Perishables"],
        help="Whether the product is perishable or not"
    )

with col2:
    st.markdown("### 🏪 Store Information")
    
    store_size = st.selectbox(
        "Store Size",
        options=["Small", "Medium", "High"],
        index=1,
        help="Size of the store based on square footage"
    )
    
    store_location = st.selectbox(
        "City Tier",
        options=["Tier 1", "Tier 2", "Tier 3"],
        index=1,
        help="Tier 1 = Metro cities, Tier 2 = Mid-size cities, Tier 3 = Small towns"
    )
    
    store_type = st.selectbox(
        "Store Type",
        options=["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
        help="Type of retail store"
    )
    
    store_age = st.slider(
        "Store Age (Years)",
        min_value=1,
        max_value=40,
        value=15,
        help="Years since the store was established"
    )

# Prediction section
st.markdown("---")
st.markdown("### 🎯 Sales Prediction")

if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
    if model is not None:
        # Prepare input data
        input_data = pd.DataFrame([{
            'Product_Weight': product_weight,
            'Product_Sugar_Content': product_sugar_content,
            'Product_Allocated_Area': product_allocated_area,
            'Product_MRP': product_mrp,
            'Store_Size': store_size,
            'Store_Location_City_Type': store_location,
            'Store_Type': store_type,
            'Product_Id_char': product_id_char,
            'Store_Age_Years': store_age,
            'Product_Type_Category': product_type_category
        }])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted Sales Revenue</p>
                <p class="prediction-value">₹{prediction:,.2f}</p>
                <p style="font-size: 0.9rem; opacity: 0.9;">per quarter for this product-store combination</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                monthly = prediction / 3
                st.metric("Monthly Estimate", f"₹{monthly:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                annual = prediction * 4
                st.metric("Annual Projection", f"₹{annual:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                # Simple confidence indicator based on typical model uncertainty
                confidence = "High" if 2000 < prediction < 6000 else "Medium"
                st.metric("Confidence Level", confidence)
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.warning("Model not loaded. Please check that the model file exists.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>📊 Built with Streamlit | 🤖 Powered by scikit-learn Random Forest</p>
    <p>Data Science Portfolio Project | Ruben Nunez | UT Austin AI/ML Program</p>
</div>
""", unsafe_allow_html=True)
