# SuperKart Sales Prediction API
# Flask backend for sales forecasting

import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Flask app initialization
superkart_api = Flask("Superkart Sales Predictor")

# Load the trained model
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    return jsonify({
        "service": "SuperKart Sales Prediction API",
        "version": "1.0",
        "endpoints": {
            "POST /v1/predict": "Predict sales for a product-store combination"
        }
    })

@superkart_api.post('/v1/predict')
def predict_sales():
    """Predict sales revenue for a product-store combination."""
    data = request.get_json()

    # Extract features (order matters for the model)
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Convert to DataFrame for prediction
    input_data = pd.DataFrame([sample])

    # Make prediction
    prediction = model.predict(input_data).tolist()[0]

    return jsonify({'Sales': prediction})


if __name__ == '__main__':
    superkart_api.run(debug=True)
