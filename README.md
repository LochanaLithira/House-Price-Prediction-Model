# 🏠 Melbourne House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.4.0-green?logo=microsoft&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-ready machine learning application that predicts Melbourne house prices using advanced feature engineering, ensemble learning, and a user-friendly web interface.

<p align="center">
  <img src="https://img.shields.io/badge/MAE-~%24150K-brightgreen" alt="MAE"/>
  <img src="https://img.shields.io/badge/R²-0.85+-blue" alt="R2 Score"/>
  <img src="https://img.shields.io/badge/Suburbs-338-purple" alt="Suburbs"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Technical Pipeline](#technical-pipeline)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an end-to-end machine learning solution for predicting residential property prices in Melbourne, Australia. The system processes raw real estate data through a sophisticated preprocessing pipeline, trains multiple regression models, and deploys the best performer via an interactive Streamlit web application.

### Key Highlights

- **Smart Location Intelligence**: Auto-fills geographic and regional data from suburb selection
- **Robust Preprocessing**: Handles missing values, outliers, and high-cardinality features
- **Production-Ready**: Cached model loading, input validation, and error handling
- **Interpretable Results**: Confidence intervals and feature importance analysis

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗺️ **Smart Suburb Lookup** | Select a suburb and automatically populate latitude, longitude, region, property count, and top agent |
| 📊 **Advanced Imputation** | KNN imputation + suburb centroid-based geographic imputation |
| 🎯 **Target Encoding** | Smoothed target encoding for high-cardinality categoricals (338 suburbs, 268 agents) |
| 📈 **Ensemble Learning** | LightGBM with 1000 estimators and hyperparameter tuning |
| ✅ **Input Validation** | Real-time validation for year, land size, and building area |
| 📉 **Confidence Estimation** | Prediction confidence intervals logged to console |
| ⚡ **Cached Loading** | Model and artifacts cached for instant predictions |

---

## 📁 Project Structure

```
Melbourne-House-Price-Prediction/
│
├── 📄 app.py                    # Streamlit web application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
│
├── 📂 data/
│   ├── 📂 raw/                  # Original dataset
│   │   └── Melbourne_housing_FULL.csv
│   └── 📂 processed/            # Transformed data (parquet)
│       ├── X_train_processed.parquet
│       ├── X_test_processed.parquet
│       ├── y_train_log.parquet
│       ├── y_test_log.parquet
│       └── y_test_real.parquet
│
├── 📂 models/
│   ├── house_price_model.joblib # Trained LightGBM model
│   └── 📂 artifacts/            # Preprocessing artifacts
│       ├── target_encoder.joblib
│       ├── ohe_encoder.joblib
│       ├── scaler_robust.joblib
│       ├── scaler_standard.joblib
│       ├── cols_robust.joblib
│       ├── cols_standard.joblib
│       ├── cols_target.joblib
│       ├── cols_ohe.joblib
│       ├── model_columns.joblib
│       └── suburb_info.joblib
│
├── 📂 notebooks/
│   ├── eda.ipynb                # Exploratory Data Analysis
│   ├── preprocessing.ipynb      # Data preprocessing pipeline
│   ├── model_training.ipynb     # Model selection & tuning
│   └── evaluation.ipynb         # Performance evaluation
│
└── 📂 reports/
    └── EDA_notes.md             # Preprocessing strategy documentation
```

---

## 📊 Dataset

**Source**: Melbourne Housing Market Dataset  
**Records**: ~34,000 properties  
**Time Period**: 2016-2018  

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Suburb` | Categorical | Suburb name (338 unique) |
| `Rooms` | Numeric | Number of rooms |
| `Type` | Categorical | h=House, u=Unit, t=Townhouse |
| `Price` | Numeric | Sale price (Target) |
| `Distance` | Numeric | Distance to CBD (km) |
| `Bathroom` | Numeric | Number of bathrooms |
| `Car` | Numeric | Car parking spots |
| `Landsize` | Numeric | Land size (sqm) |
| `BuildingArea` | Numeric | Building area (sqm) |
| `YearBuilt` | Numeric | Year of construction |
| `Regionname` | Categorical | General region |
| `Propertycount` | Numeric | Properties in suburb |
| `Lattitude` | Numeric | Geographic latitude |
| `Longtitude` | Numeric | Geographic longitude |
| `SellerG` | Categorical | Real estate agent (268 unique) |

---

## 🔧 Technical Pipeline

### Preprocessing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Data ──► Data Cleaning ──► Train/Test Split ──► Feature Engineering    │
│                    │                                         │               │
│                    ▼                                         ▼               │
│            • Remove duplicates                    • Extract SoldYear         │
│            • Fix impossible values                • Group rare regions       │
│            • Standardize text                                                │
│                                                                              │
│  ──► Imputation ──► Log Transform ──► Scaling ──► Encoding ──► Alignment    │
│          │               │              │            │             │         │
│          ▼               ▼              ▼            ▼             ▼         │
│    • KNN (5 neighbors)  • np.log1p   • Robust    • One-Hot    • Reindex     │
│    • Suburb centroids   • 6 features • Standard  • Target     • Fill gaps   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Transformation Details

| Stage | Features | Method |
|-------|----------|--------|
| **Log Transform** | Distance, Bathroom, Car, Landsize, BuildingArea, Propertycount | `np.log1p()` |
| **RobustScaler** | Rooms, Distance, Bathroom, Car, Landsize, BuildingArea, Propertycount | Median/IQR based |
| **StandardScaler** | YearBuilt, Lattitude, Longtitude | Mean/Std based |
| **One-Hot Encoding** | Type, Regionname | Sparse binary columns |
| **Target Encoding** | Suburb, SellerG | Smoothed mean encoding (smoothing=10) |

---

## 🤖 Model Architecture

### Model Selection Process

| Model | CV MAE (Log) | Status |
|-------|--------------|--------|
| Linear Regression | Baseline | ❌ |
| Random Forest | Good | ❌ |
| XGBoost | Better | ❌ |
| **LightGBM** | **Best** | ✅ Selected |

### Final Model Configuration

```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
```

### Hyperparameter Tuning

- **Method**: RandomizedSearchCV
- **CV Folds**: 3
- **Iterations**: 20 random combinations
- **Scoring**: Negative Mean Absolute Error

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/melbourne-house-price-prediction.git
   cd melbourne-house-price-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   
   # Windows
   .\env\Scripts\activate
   
   # macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   ```
   Open http://localhost:8501 in your browser
   ```

---

## 💻 Usage

### Web Application

1. **Select Location**: Choose a suburb from the dropdown (auto-fills geographic data)
2. **Enter Property Details**: 
   - Number of rooms, bathrooms, car spots
   - Property type (House/Unit/Townhouse)
   - Year built, distance to CBD
   - Land size and building area
3. **Get Prediction**: Click "Estimate Price" for instant valuation

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Run in order: eda.ipynb → preprocessing.ipynb → model_training.ipynb → evaluation.ipynb
```

### Retraining the Model

```bash
# 1. Place new data in data/raw/
# 2. Run preprocessing notebook (generates new artifacts)
# 3. Run model_training notebook (saves new model)
# 4. Restart Streamlit app
```

---

## 📡 API Reference

### Prediction Pipeline (app.py)

```python
# Input Features Required
input_data = {
    'Suburb': str,           # Must be in suburb_info.joblib
    'Rooms': int,            # 1-8
    'Type': str,             # 'h', 'u', or 't'
    'SellerG': str,          # Auto-filled from suburb
    'Distance': float,       # 0-60 km
    'Bathroom': int,         # 1-5
    'Car': int,              # 0-5
    'Landsize': float,       # > 0 sqm
    'BuildingArea': float,   # > 0 sqm
    'YearBuilt': int,        # 1850-current year
    'Regionname': str,       # Auto-filled from suburb
    'Propertycount': int,    # Auto-filled from suburb
    'Lattitude': float,      # Auto-filled from suburb
    'Longtitude': float,     # Auto-filled from suburb
    'SoldYear': int          # Current year
}
```

### Artifacts Structure

| Artifact | Type | Shape/Keys |
|----------|------|------------|
| `house_price_model.joblib` | LGBMRegressor | Trained model |
| `suburb_info.joblib` | dict | 338 suburbs → {Lat, Long, Region, Count, Seller} |
| `model_columns.joblib` | list | 22 feature names |
| `target_encoder.joblib` | TargetEncoder | Fitted on Suburb, SellerG |
| `ohe_encoder.joblib` | OneHotEncoder | Fitted on Type, Regionname |

---

## 📈 Performance Metrics

### Test Set Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~$150,000 | Average prediction error |
| **RMSE** | ~$220,000 | Root mean squared error |
| **R²** | ~0.85 | 85% variance explained |

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Suburb (encoded) | High |
| 2 | BuildingArea | High |
| 3 | Distance | High |
| 4 | Rooms | Medium |
| 5 | YearBuilt | Medium |
| 6 | Landsize | Medium |
| 7 | Regionname | Medium |
| 8 | Type | Medium |
| 9 | Bathroom | Low |
| 10 | Car | Low |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11 |
| **ML Framework** | scikit-learn, LightGBM, XGBoost |
| **Data Processing** | pandas, NumPy |
| **Encoding** | category_encoders (TargetEncoder) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Serialization** | joblib, pyarrow (parquet) |
| **Notebooks** | Jupyter, JupyterLab |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Update notebooks if changing pipeline
- Test with `streamlit run app.py` before committing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Melbourne Housing Dataset from Kaggle
- scikit-learn and LightGBM communities
- Streamlit for the amazing web framework

---

<p align="center">
  <b>Built with ❤️ for Melbourne Property Insights</b>
</p>

<p align="center">
  <a href="#-melbourne-house-price-prediction">Back to Top ↑</a>
</p>

