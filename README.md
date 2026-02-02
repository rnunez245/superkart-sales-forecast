# 🛒 SuperKart Sales Forecasting

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/rnunez245/superkart-frontend)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**ML-powered sales revenue prediction for retail inventory optimization**

🔗 **[Try the Live Demo](https://huggingface.co/spaces/rnunez245/superkart-frontend)**

---

## 📋 Business Problem

SuperKart, a retail chain operating supermarkets and food marts across various tier cities, needs to accurately forecast sales revenue to:
- Optimize inventory management
- Plan regional sales strategies  
- Make informed procurement decisions
- Reduce stockouts and overstock situations

## 🎯 Solution

Built an end-to-end ML pipeline that predicts quarterly sales revenue based on product attributes and store characteristics, deployed as a microservices architecture with Flask API backend and Streamlit frontend.

### Model Performance
| Metric | Training | Testing |
|--------|----------|---------|
| R² Score | 0.65 | 0.63 |
| RMSE | 628 | 590 |
| MAPE | 17.3% | 17.8% |

## 🔧 Features Used

**Product Features:**
- Weight, MRP, Display Area Ratio
- Sugar Content (Low/Regular/None)
- Product Category (Food/Drinks/Non-Consumables)
- Perishability

**Store Features:**
- Size (Small/Medium/High)
- City Tier (1/2/3)
- Store Type (Supermarket/Departmental/Food Mart)
- Store Age

## 🏗️ Architecture

```
┌─────────────────┐     HTTP/JSON      ┌─────────────────┐
│   Streamlit     │ ──────────────────▶│   Flask API     │
│   Frontend      │                    │   Backend       │
│   (User UI)     │ ◀────────────────  │   (ML Model)    │
└─────────────────┘    Predictions     └─────────────────┘
        │                                      │
        ▼                                      ▼
   HF Spaces                              HF Spaces
   (Docker)                               (Docker)
```

## 🛠️ Technical Stack

- **ML Framework:** scikit-learn (Random Forest Regressor)
- **Backend:** Flask + Gunicorn
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces (Docker)
- **Data Processing:** pandas, NumPy
- **Version Control:** Git/GitHub

## 📁 Project Structure

```
superkart-sales-forecast/
├── app.py                 # Standalone Streamlit app (alternative)
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             
├── data/
│   └── SuperKart.csv      # Dataset (not in repo)
├── model/
│   └── superkart_model.joblib  # Trained model
├── notebooks/
│   └── sales_forecast.ipynb    # EDA & experimentation
└── deploy/
    ├── backend/           # Flask API deployment
    │   ├── app.py
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── superkart_model.joblib
    └── frontend/          # Streamlit UI deployment
        ├── app.py
        ├── Dockerfile
        └── requirements.txt
```

## 🚀 Deployment

### Live Demo
- **Frontend:** [huggingface.co/spaces/rnunez245/superkart-frontend](https://huggingface.co/spaces/rnunez245/superkart-frontend)
- **Backend API:** [huggingface.co/spaces/rnunez245/superkart-backend](https://huggingface.co/spaces/rnunez245/superkart-backend)

### Local Development
```bash
# Clone repository
git clone https://github.com/rnunez245/superkart-sales-forecast.git
cd superkart-sales-forecast

# Install dependencies
pip install -r requirements.txt

# Train model (requires data/SuperKart.csv)
python train_model.py

# Run standalone app
streamlit run app.py
```

## 📊 Key Insights

1. **Price & Weight Drive Sales**: Products with higher MRP and weight generate more revenue (correlation ~0.79)
2. **Store Type Matters**: Supermarket Type 2 stores significantly outperform other formats
3. **Tier 2 Cities Lead**: Tier 2 city locations show strongest sales performance
4. **Low Sugar Preference**: Low-sugar products dominate revenue share

## 📈 Business Recommendations

1. **Product Strategy**: Prioritize premium, heavier products in high-performing categories
2. **Store Expansion**: Focus new stores on Tier 2 cities with Supermarket Type 2 format
3. **Inventory Optimization**: Use predictions for dynamic stock allocation
4. **Health Trends**: Maintain focus on low-sugar products

## 👨‍💻 Author

**Ruben Nunez**  
UT Austin Post Graduate Program in AI/ML  
🏆 **Ranked Top 5** in class | GPA: 4.28

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/rnunez245)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/rnunez245)

---

*Part of the UT Austin AI/ML Program Business Case Studies*
