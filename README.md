# 🏠 Melbourne House Price Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![Status](https://img.shields.io/badge/Status-Completed-green)

An end-to-end Machine Learning solution to predict housing prices in Melbourne, Australia. This project demonstrates a complete lifecycle from Exploratory Data Analysis (EDA) and advanced preprocessing to model tuning and deployment via a Streamlit web application.

## 🚀 Key Features & Advanced Techniques

This project goes beyond standard implementations by utilizing domain-specific logic and advanced engineering techniques:

* **📍 Geographic-Aware Imputation:** Uses **Suburb Centroid Imputation** to fill missing location data based on reliable anchors (≥3 houses), preserving spatial relationships.
* **🎯 Target Encoding with Smoothing:** Handles high-cardinality features (Suburb, SellerG) effectively, using smoothing (10.0) to prevent overfitting on rare categories.
* **⚖️ Dual-Scaling Strategy:** Applies `RobustScaler` for outlier-prone features and `StandardScaler` for normally distributed features.
* **🧠 Smart Suburb Lookup:** The web app automatically fills 5 related fields (Region, Lat/Lon, etc.) when a user selects a suburb, improving user experience and data accuracy.
* **🔮 Confidence Estimation:** Calculates prediction confidence intervals during inference using iteration sampling.

## 🛠️ Tech Stack

* **Core:** Python, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Missingno
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost
* **Deployment:** Streamlit
* **Data Management:** Joblib (Serialization), Parquet

## 📂 Project Structure

```text
House-Price-Prediction-Model/
├── app.py                      # Streamlit web application entry point
├── requirements.txt            # Project dependencies
├── data/
│   ├── raw/                    # Original Melbourne_housing_FULL.csv
│   └── processed/              # Processed parquet files for training
├── models/
│   ├── house_price_model.joblib # Final trained LightGBM model
│   └── artifacts/              # 10+ preprocessing artifacts (scalers, encoders)
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis & Cleaning
│   ├── preprocessing.ipynb     # Feature engineering & Transformation pipeline
│   ├── model_training.ipynb    # Model comparison & Hyperparameter tuning
│   └── evaluation.ipynb        # Performance metrics & Residual analysis
└── reports/                    # Documentation and notes
