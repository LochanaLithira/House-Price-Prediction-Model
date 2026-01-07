<div align="center">

# 🏠 Melbourne House Price Prediction

### Production-Ready ML Pipeline for Real Estate Valuation

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*An end-to-end machine learning solution for estimating residential property prices in Melbourne, Australia.*

[**Live Demo**](#-quick-start) · [**Documentation**](#-data-science-workflow) · [**Report Bug**](../../issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Quick Start](#-quick-start)
- [Data Science Workflow](#-data-science-workflow)
- [Model Performance](#-model-performance)
- [Application Features](#-application-features)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Real estate valuation is a complex problem influenced by numerous location-specific and property-specific factors. This project delivers a **production-ready machine learning pipeline** that provides instant property valuations based on historical Melbourne housing market data.

### Business Value
- **Instant Valuations**: Get property estimates in milliseconds
- **Data-Driven Decisions**: Powered by 20,000+ historical transactions
- **Explainable Results**: Transparent feature importance analysis
- **Deployment Ready**: Complete with web interface and API-ready architecture

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Robust Data Pipeline** | Handles missing values via KNN & Centroid imputation with intelligent fallbacks |
| 📊 **Advanced Feature Engineering** | Log transforms, Target Encoding, and RobustScaler for outlier resistance |
| 🤖 **Multi-Model Benchmarking** | Compares Linear Regression, Random Forest, XGBoost, and LightGBM |
| ⚡ **Hyperparameter Tuning** | Automated optimization via `RandomizedSearchCV` with 3-fold CV |
| 🎨 **Interactive Web Dashboard** | User-friendly Streamlit interface with real-time predictions |
| 💾 **Artifact Persistence** | Production-ready model serialization with complete preprocessing pipeline |
| 📈 **Confidence Intervals** | Ensemble-based prediction uncertainty estimation |

---

## 🏆 Advanced Techniques & Methodologies

> **What sets this project apart:** This section highlights the sophisticated, production-grade techniques implemented that go beyond typical ML tutorials.

### 🛡️ Data Leakage Prevention

```
❌ WRONG: Fit encoders on ALL data → Transform → Split
✅ CORRECT: Split FIRST → Fit on Train ONLY → Transform both
```

| Strategy | Implementation |
|----------|----------------|
| **Split-First Protocol** | All transformations fitted exclusively on training data |
| **Temporal Awareness** | Extracted `SoldYear` to capture market inflation effects |
| **Strict Isolation** | Test set never seen during any preprocessing step |

### 🎯 Intelligent Missing Value Imputation

Instead of dropping 50%+ of data or using naive mean/median fills:

| Technique | Applied To | Why It's Better |
|-----------|------------|-----------------|
| **KNN Imputation (k=5)** | `BuildingArea`, `YearBuilt`, `Car`, `Bathroom`, `Landsize` | Leverages similarity between properties — a 3-bedroom house in Richmond likely has similar building area to other 3-bedroom Richmond houses |
| **Spatial Centroid Imputation** | `Latitude`, `Longitude` | Custom algorithm: computes median coordinates per suburb from training data, with global median fallback for rare suburbs |
| **Mode/Median with Validation** | `Regionname`, `Propertycount` | Simple but applied AFTER split to prevent leakage |

```python
# Spatial Centroid Imputation (Rare Technique)
# Instead of dropping rows with missing coordinates:

# 1. Compute suburb centroids from TRAINING data only
suburb_coords = X_train.groupby('Suburb')[['Lattitude', 'Longtitude']].median()

# 2. Filter unreliable suburbs (< 3 samples)
reliable_suburbs = suburb_counts[suburb_counts >= 3].index

# 3. Apply to both sets with global fallback
X_train['Lattitude'].fillna(X_train['Suburb'].map(suburb_coords['Lattitude']))
```

### 📊 Dual Scaling Strategy

> **Key Insight:** Not all features should be scaled the same way.

| Scaler | Features | Rationale |
|--------|----------|-----------|
| **RobustScaler** | `Rooms`, `Distance`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `Propertycount` | Uses IQR instead of variance — **immune to outliers**. A $9M mansion won't skew the scale. |
| **StandardScaler** | `YearBuilt`, `Latitude`, `Longitude` | Normally distributed features without extreme outliers |

```python
# Why RobustScaler for Landsize?
# Landsize has outliers: 99th percentile = 1,200 sqm, max = 400,000+ sqm
# StandardScaler would compress 99% of data into tiny range
# RobustScaler ignores outliers: centers on median, scales by IQR
```

### 🔐 Target Encoding with Smoothing

> **Problem:** `Suburb` has 300+ unique values — One-Hot Encoding creates 300 sparse columns.

| Approach | Issues |
|----------|--------|
| ❌ One-Hot Encoding | Dimensionality explosion, sparse matrix, overfitting |
| ❌ Label Encoding | Imposes false ordinal relationship |
| ✅ **Target Encoding + Smoothing** | Maps each suburb to average price, regularized |

```python
# Target Encoding with Smoothing (Rare Technique)
from category_encoders import TargetEncoder

encoder = TargetEncoder(cols=['Suburb', 'SellerG'], smoothing=10.0)

# smoothing=10 prevents overfitting on rare suburbs:
# If "NewSuburb" has only 2 sales, it won't overfit to those 2 prices
# Instead, it blends toward the global mean price
```

**Smoothing Formula:**
$$\text{Encoded Value} = \frac{n \cdot \bar{x}_{\text{category}} + m \cdot \bar{x}_{\text{global}}}{n + m}$$

Where $n$ = category count, $m$ = smoothing factor, $\bar{x}$ = mean target

### 📈 Ensemble-Based Confidence Intervals

> **Beyond point predictions:** Quantify uncertainty using tree ensemble variance.

```python
# For Random Forest / XGBoost with multiple trees:
if hasattr(model, 'estimators_'):
    # Get prediction from EACH tree
    tree_preds = [np.expm1(tree.predict(df)[0]) for tree in model.estimators_]
    
    # Standard deviation = prediction uncertainty
    std = np.std(tree_preds)
    
    # Confidence interval
    confidence_range = (price - std, price + std)
    
    # Coefficient of variation → confidence percentage
    confidence_pct = max(0, min(100, 100 - (std/price)*100))
```

### 🔄 Complete Artifact Serialization

> **Production Pattern:** Serialize the entire preprocessing pipeline, not just the model.

```
models/artifacts/
├── target_encoder.joblib    # Fitted on training targets
├── ohe_encoder.joblib       # Category mappings
├── scaler_robust.joblib     # IQR statistics from training
├── scaler_standard.joblib   # Mean/std from training
├── suburb_info.joblib       # Suburb → (lat, long, region) lookup
├── cols_robust.joblib       # Which columns use RobustScaler
├── cols_standard.joblib     # Which columns use StandardScaler
├── cols_target.joblib       # Target encoding columns
├── cols_ohe.joblib          # One-hot encoding columns
└── model_columns.joblib     # Exact column order for prediction
```

**Why this matters:**
- ✅ New predictions use exact same transformations as training
- ✅ No "training-serving skew" in production
- ✅ Model can be deployed without access to training data

### 🧮 Log Transform Strategy

| Transform | Applied To | Purpose |
|-----------|------------|---------|
| `np.log1p(x)` | `Price` (target) | Normalize right-skewed distribution, improve model assumptions |
| `np.log1p(x)` | `Distance`, `Landsize`, `BuildingArea`, `Propertycount` | Handle extreme skewness (Landsize skew: 40+) |
| `np.expm1(pred)` | Predictions | Inverse transform to real dollar values |

```python
# Why log1p instead of log?
# log(0) = undefined ❌
# log1p(0) = log(1) = 0 ✅  (handles zero values safely)
```

---

## 🛠 Tech Stack

### Core ML/Data Science
```
pandas==2.0.3          # Data manipulation & analysis
numpy==1.26.4          # Numerical computing
scikit-learn==1.3.0    # ML algorithms & preprocessing
xgboost==1.7.6         # Gradient boosting
lightgbm==4.4.0        # Light gradient boosting
category_encoders      # Target encoding for high-cardinality features
```

### Visualization & Analysis
```
matplotlib==3.8.0      # Static visualizations
seaborn==0.12.2        # Statistical graphics
plotly==6.1.0          # Interactive plots
missingno              # Missing data visualization
statsmodels            # Statistical modeling
```

### Deployment & Infrastructure
```
streamlit              # Web application framework
joblib==1.3.2          # Model serialization
pyarrow                # Parquet file support
jupyterlab==4.3.0      # Development environment
```

---

## 📁 Project Architecture

```
House-Price-Prediction-Model/
│
├── 📱 app.py                           # Streamlit web application
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                        # Project documentation
│
├── 📊 data/
│   ├── raw/
│   │   └── Melbourne_housing_FULL.csv  # Original dataset (34,857 records)
│   └── processed/
│       ├── X_train_processed.parquet   # Preprocessed training features
│       ├── X_test_processed.parquet    # Preprocessed test features
│       ├── y_train_log.parquet         # Log-transformed training targets
│       └── y_test_log.parquet          # Log-transformed test targets
│
├── 🤖 models/
│   ├── house_price_model.joblib        # Final production model
│   └── artifacts/
│       ├── target_encoder.joblib       # Suburb/Seller target encoder
│       ├── ohe_encoder.joblib          # One-hot encoder (Type, Region)
│       ├── scaler_robust.joblib        # RobustScaler for outlier features
│       ├── scaler_standard.joblib      # StandardScaler for normal features
│       ├── suburb_info.joblib          # Suburb metadata dictionary
│       ├── cols_robust.joblib          # Column mapping for RobustScaler
│       ├── cols_standard.joblib        # Column mapping for StandardScaler
│       ├── cols_target.joblib          # Target encoding columns
│       ├── cols_ohe.joblib             # One-hot encoding columns
│       └── model_columns.joblib        # Final feature column order
│
├── 📓 notebooks/
│   ├── eda.ipynb                       # Exploratory Data Analysis (33 cells)
│   ├── preprocessing.ipynb             # Data cleaning & feature engineering (31 cells)
│   ├── model_training.ipynb            # Model selection & hyperparameter tuning (8 cells)
│   └── evaluation.ipynb                # Performance analysis & diagnostics (9 cells)
│
├── 📝 reports/
│   └── EDA_notes.md                    # Data insights & preprocessing strategy
│
└── 🔧 src/                             # Source modules (extensible)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/House-Price-Prediction-Model.git
cd House-Price-Prediction-Model

# 2. Create virtual environment
python -m venv env

# 3. Activate environment
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`

### Run Jupyter Notebooks

```bash
jupyter lab
```

Navigate to `notebooks/` to explore the data science workflow.

---

## 🧠 Data Science Workflow

### Phase 1: Exploratory Data Analysis
📍 **Notebook**: `notebooks/eda.ipynb`

| Analysis Type | Findings |
|---------------|----------|
| **Data Quality** | 34,857 records, 21 features, 1 duplicate removed |
| **Missing Values** | BuildingArea (50%+), YearBuilt (50%+), Lat/Long (7%) |
| **Target Distribution** | Right-skewed ($300K - $9M), log-transform required |
| **Key Correlations** | Rooms ↔ Price (+0.49), Distance ↔ Price (-0.32) |
| **Outlier Detection** | Valid high-value properties in premium suburbs retained |

**Key Visualizations:**
- Missing value matrix (missingno)
- Price distribution by property type
- Geographic heatmaps (Latitude/Longitude)
- Correlation matrices with VIF analysis

### Phase 2: Data Preprocessing
📍 **Notebook**: `notebooks/preprocessing.ipynb`

#### Data Cleaning Pipeline

```
Raw Data (34,857)
    │
    ▼
┌─────────────────────────────────────┐
│  CLEANING                           │
│  • Remove duplicates                │
│  • Fix inconsistent text casing     │
│  • Handle impossible values:        │
│    - BuildingArea < 5 → NaN         │
│    - YearBuilt > 2026 → NaN         │
│    - Bathroom = 0 → NaN             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  FEATURE ENGINEERING                │
│  • Extract SoldYear from Date       │
│  • Group rare regions → "Regional   │
│    Victoria"                        │
│  • Drop: Address, Postcode,         │
│    CouncilArea, Method, Bedroom2    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  IMPUTATION                         │
│  • KNN Imputer (k=5):               │
│    Car, Landsize, BuildingArea,     │
│    YearBuilt, Bathroom              │
│  • Suburb Centroid:                 │
│    Lattitude, Longtitude            │
│  • Mode/Median:                     │
│    Regionname, Propertycount        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  TRANSFORMATION                     │
│  • Log1p: Price, Distance,          │
│    Bathroom, Car, Landsize,         │
│    BuildingArea, Propertycount      │
│  • RobustScaler: Outlier features   │
│  • StandardScaler: Normal features  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  ENCODING                           │
│  • One-Hot: Type, Regionname        │
│  • Target Encoding (smoothing=10):  │
│    Suburb, SellerG                  │
└─────────────────────────────────────┘
    │
    ▼
Clean Data (27,247 records, 80/20 split)
```

### Phase 3: Model Training
📍 **Notebook**: `notebooks/model_training.ipynb`

#### Algorithm Comparison (5-Fold CV)

| Model | CV MAE (Log) | Training Time |
|-------|-------------|---------------|
| Linear Regression | Baseline | Fast |
| Random Forest | Good | Medium |
| **XGBoost** | **Best** | Medium |
| LightGBM | Excellent | Fast |

#### Hyperparameter Search Space

```python
param_grids = {
    "XGBoost": {
        'n_estimators': [500, 1000, 2000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    },
    "LightGBM": {
        'n_estimators': [500, 1000, 2000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'subsample': [0.7, 0.8, 0.9]
    }
}
```

### Phase 4: Model Evaluation
📍 **Notebook**: `notebooks/evaluation.ipynb`

**Diagnostic Plots:**
- Actual vs Predicted scatter plot
- Residual analysis (homoscedasticity check)
- Error distribution histogram
- Feature importance ranking (Top 20)
- Worst predictions analysis

---

## 📈 Model Performance

### Final Test Set Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~$150,000 | Average prediction error in dollars |
| **RMSE** | ~$220,000 | Penalizes large errors more heavily |
| **R²** | ~0.75 | Model explains 75% of price variance |

### Top Predictive Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `Suburb` (Target Encoded) | High |
| 2 | `BuildingArea` | High |
| 3 | `Rooms` | Medium |
| 4 | `Distance` (to CBD) | Medium |
| 5 | `YearBuilt` | Medium |

---

## 🎨 Application Features

### Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Suburb | Dropdown | 300+ options | Auto-fills lat/long, region, property count |
| Rooms | Slider | 1-8 | Number of bedrooms |
| Bathrooms | Slider | 1-5 | Number of bathrooms |
| Car Spots | Slider | 0-5 | Parking spaces |
| Property Type | Select | House/Unit/Townhouse | Building type |
| Year Built | Input | 1850-2026 | Construction year |
| Distance to CBD | Input | 0-60 km | Distance to city center |
| Land Size | Input | sqm | Total land area |
| Building Area | Input | sqm | Floor area |

### Prediction Pipeline

```
User Input
    │
    ▼
┌────────────────────────┐
│  Input Validation      │
│  • Future year check   │
│  • Positive area check │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Suburb Lookup         │
│  • Auto-fill lat/long  │
│  • Auto-fill region    │
│  • Auto-fill seller    │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Preprocessing         │
│  • Log transform       │
│  • Apply scalers       │
│  • Encode features     │
│  • Column alignment    │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│  Model Prediction      │
│  • Log prediction      │
│  • Inverse transform   │
│  • Confidence interval │
└────────────────────────┘
    │
    ▼
Estimated Price ($)
```

---

## 📚 API Reference

### Loading Model Artifacts

```python
import joblib
import pandas as pd
import numpy as np

# Define paths
ART_PATH = 'models/artifacts/'
MODEL_PATH = 'models/house_price_model.joblib'

# Load model
model = joblib.load(MODEL_PATH)

# Load preprocessing artifacts
suburb_info = joblib.load(f'{ART_PATH}suburb_info.joblib')
te = joblib.load(f'{ART_PATH}target_encoder.joblib')
ohe = joblib.load(f'{ART_PATH}ohe_encoder.joblib')
s_rob = joblib.load(f'{ART_PATH}scaler_robust.joblib')
s_std = joblib.load(f'{ART_PATH}scaler_standard.joblib')
model_cols = joblib.load(f'{ART_PATH}model_columns.joblib')
```

### Making Predictions

```python
# Create input DataFrame
input_data = {
    'Suburb': 'Richmond',
    'Rooms': 3,
    'Type': 'h',
    'Distance': 5.0,
    'Bathroom': 2,
    'Car': 1,
    'Landsize': 450,
    'BuildingArea': 120,
    'YearBuilt': 1970,
    # ... additional features
}

df = pd.DataFrame([input_data])

# Apply preprocessing pipeline
# ... (see app.py for full implementation)

# Predict
log_pred = model.predict(df)[0]
price = np.expm1(log_pred)

print(f"Estimated Price: ${price:,.0f}")
```

---

## 🗂 Dataset Information

### Source
**Melbourne Housing Market** dataset containing historical property sales data.

### Features (21 Original)

| Feature | Type | Description |
|---------|------|-------------|
| `Suburb` | Categorical | Suburb name |
| `Address` | Categorical | Full street address (dropped) |
| `Rooms` | Numeric | Number of rooms |
| `Type` | Categorical | h=house, u=unit, t=townhouse |
| `Price` | Numeric | Sale price (target) |
| `Method` | Categorical | Sale method (dropped) |
| `SellerG` | Categorical | Real estate agent |
| `Date` | DateTime | Sale date |
| `Distance` | Numeric | Distance from CBD (km) |
| `Postcode` | Numeric | Postal code (dropped) |
| `Bedroom2` | Numeric | Bedroom count (dropped, redundant) |
| `Bathroom` | Numeric | Bathroom count |
| `Car` | Numeric | Car parking spots |
| `Landsize` | Numeric | Land size (sqm) |
| `BuildingArea` | Numeric | Building size (sqm) |
| `YearBuilt` | Numeric | Year of construction |
| `CouncilArea` | Categorical | Council jurisdiction (dropped) |
| `Lattitude` | Numeric | Geographic latitude |
| `Longtitude` | Numeric | Geographic longitude |
| `Regionname` | Categorical | General region |
| `Propertycount` | Numeric | Properties in suburb |

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
- Add docstrings to all functions
- Update tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Melbourne Housing Dataset contributors
- scikit-learn, XGBoost, and LightGBM development teams
- Streamlit for the amazing web framework

---

<div align="center">

**Built with ❤️ for the Data Science Community**

[⬆ Back to Top](#-melbourne-house-price-prediction)

</div>
