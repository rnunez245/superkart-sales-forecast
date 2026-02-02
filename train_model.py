"""
SuperKart Model Training Script
================================
Trains a Random Forest model and saves it for deployment.

Usage:
    python train_model.py

Requirements:
    - Place SuperKart.csv in the data/ folder
    - Run this script to generate model/superkart_model.joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os

def load_and_preprocess_data(filepath='data/SuperKart.csv'):
    """Load and preprocess the SuperKart dataset."""
    print("📁 Loading data...")
    df = pd.read_csv(filepath)
    
    # Feature engineering
    print("🔧 Engineering features...")
    
    # Extract product category from ID (first 2 characters)
    df['Product_Id_char'] = df['Product_Id'].str[:2]
    
    # Calculate store age
    current_year = 2024  # Reference year
    df['Store_Age_Years'] = current_year - df['Store_Establishment_Year']
    
    # Create product type category (Perishables vs Non-Perishables)
    perishables = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Seafood', 'Bread', 'Breakfast', 'Frozen Foods']
    df['Product_Type_Category'] = df['Product_Type'].apply(
        lambda x: 'Perishables' if x in perishables else 'Non Perishables'
    )
    
    return df

def prepare_features(df):
    """Prepare features and target for training."""
    # Define feature columns
    feature_cols = [
        'Product_Weight',
        'Product_Sugar_Content',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Id_char',
        'Store_Age_Years',
        'Product_Type_Category'
    ]
    
    X = df[feature_cols]
    y = df['Product_Store_Sales_Total']
    
    return X, y

def build_pipeline():
    """Build the preprocessing and model pipeline."""
    # Define categorical columns for one-hot encoding
    categorical_cols = [
        'Product_Sugar_Content',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Id_char',
        'Product_Type_Category'
    ]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'  # Keep numeric columns as-is
    )
    
    # Create pipeline with tuned Random Forest
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            max_features='sqrt',
            random_state=1,
            n_jobs=-1
        ))
    ])
    
    return pipeline

def train_and_evaluate(X, y):
    """Train the model and evaluate performance."""
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    
    print("🏋️ Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Model Performance:")
    print("=" * 50)
    
    # Training metrics
    y_train_pred = pipeline.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    
    # Testing metrics
    y_test_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    print(f"\nTest R² Score: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAPE: {test_mape*100:.2f}%")
    print("=" * 50)
    
    return pipeline

def save_model(pipeline, filepath='model/superkart_model.joblib'):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"\n✅ Model saved to {filepath}")
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📦 Model size: {size_mb:.2f} MB")

def main():
    print("🚀 SuperKart Model Training")
    print("=" * 50)
    
    # Check if data exists
    data_path = 'data/SuperKart.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please place the SuperKart.csv file in the data/ folder.")
        return
    
    # Load and preprocess
    df = load_and_preprocess_data(data_path)
    print(f"✅ Loaded {len(df)} records")
    
    # Prepare features
    X, y = prepare_features(df)
    print(f"✅ Prepared {X.shape[1]} features")
    
    # Train and evaluate
    pipeline = train_and_evaluate(X, y)
    
    # Save model
    save_model(pipeline)
    
    print("\n🎉 Training complete! Ready for deployment.")

if __name__ == "__main__":
    main()
