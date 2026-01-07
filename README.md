<div align="center">

# 🏠 Melbourne House Price Prediction

### Production-Ready ML Pipeline for Real Estate Valuation

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

*An end-to-end machine learning solution for estimating residential property prices in Melbourne, Australia.*


</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Advanced Techniques](#-advanced-techniques--methodologies)
- [Project Architecture](#-project-architecture)
- [Quick Start](#-quick-start)
- [Data Science Workflow](#-data-science-workflow)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Dataset](#-dataset-information)
- [Contributing](#-contributing)

---

## 🎯 Overview

Real estate valuation is a complex problem influenced by numerous location-specific and property-specific factors. This project delivers a **production-ready machine learning pipeline** that provides instant property valuations based on historical Melbourne housing market data.

### Business Value
| Benefit | Description |
|---------|-------------|
| ⚡ **Instant Valuations** | Property estimates in milliseconds |
| 📊 **Data-Driven** | Powered by 27,000+ historical transactions |
| 🔍 **Explainable** | Transparent feature importance analysis |
| 🚀 **Production-Ready** | Complete web interface with serialized pipeline |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Robust Data Pipeline** | KNN & Spatial Centroid imputation for 50%+ missing data |
| 📊 **Advanced Feature Engineering** | Dual scaling strategy + Target Encoding with smoothing |
| 🤖 **Multi-Model Benchmarking** | Linear Regression, Random Forest, XGBoost, LightGBM |
| ⚡ **Hyperparameter Tuning** | `RandomizedSearchCV` with 3-fold cross-validation |
| 🎨 **Interactive Dashboard** | Streamlit UI with real-time predictions |
| 📈 **Confidence Intervals** | Ensemble-based prediction uncertainty estimation |

---

## 🏆 Advanced Techniques & Methodologies

> **What sets this project apart:** Production-grade techniques that go beyond typical ML tutorials.

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

---

### 🎯 Intelligent Missing Value Imputation

Instead of dropping 50%+ of data or using naive mean/median fills:

| Technique | Applied To | Why It's Better |
|-----------|------------|-----------------|
| **KNN Imputation (k=5)** | `BuildingArea`, `YearBuilt`, `Car`, `Bathroom`, `Landsize` | Leverages property similarity — a 3-bedroom house in Richmond likely has similar specs to other 3-bedroom Richmond houses |
| **Spatial Centroid Imputation** | `Latitude`, `Longitude` | Custom algorithm using suburb medians with global fallback |

```python
# Spatial Centroid Imputation (Custom Technique)
# 1. Compute suburb centroids from TRAINING data only
suburb_coords = X_train.groupby('Suburb')[['Lattitude', 'Longtitude']].median()

# 2. Filter unreliable suburbs (< 3 samples)
reliable_suburbs = suburb_counts[suburb_counts >= 3].index

# 3. Apply with global median fallback for rare suburbs
X_train['Lattitude'].fillna(X_train['Suburb'].map(suburb_coords['Lattitude']))
```

---

### 📊 Dual Scaling Strategy

> **Key Insight:** Not all features should be scaled the same way.

| Scaler | Features | Rationale |
|--------|----------|-----------|
| **RobustScaler** | `Rooms`, `Distance`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `Propertycount` | Uses IQR — **immune to outliers**. A $9M mansion won't skew the scale. |
| **StandardScaler** | `YearBuilt`, `Latitude`, `Longitude` | Normally distributed features without extreme outliers |

```python
# Why RobustScaler for Landsize?
# Landsize: 99th percentile = 1,200 sqm, max = 400,000+ sqm
# StandardScaler compresses 99% of data into tiny range
# RobustScaler: centers on median, scales by IQR (ignores outliers)
```

---

### 🔐 Target Encoding with Smoothing

> **Problem:** `Suburb` has 300+ unique values — One-Hot creates 300 sparse columns.

| Approach | Issues |
|----------|--------|
| ❌ One-Hot Encoding | Dimensionality explosion, sparse matrix |
| ❌ Label Encoding | Imposes false ordinal relationship |
| ✅ **Target Encoding + Smoothing** | Maps suburb to regularized average price |

```python
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['Suburb', 'SellerG'], smoothing=10.0)

# smoothing=10: prevents overfitting on rare suburbs
# "NewSuburb" with 2 sales blends toward global mean
```

**Smoothing Formula:**
$$\text{Encoded Value} = \frac{n \cdot \bar{x}_{\text{category}} + m \cdot \bar{x}_{\text{global}}}{n + m}$$

Where $n$ = category count, $m$ = smoothing factor, $\bar{x}$ = mean target

---

### 📈 Ensemble-Based Confidence Intervals

> **Beyond point predictions:** Quantify uncertainty using tree ensemble variance.

```python
if hasattr(model, 'estimators_'):
    # Get prediction from EACH tree
    tree_preds = [np.expm1(tree.predict(df)[0]) for tree in model.estimators_]
    std = np.std(tree_preds)  # Standard deviation = uncertainty
    confidence_range = (price - std, price + std)
```

---

### 🧮 Log Transform Strategy

| Transform | Purpose |
|-----------|---------|
| `np.log1p(Price)` | Normalize right-skewed target distribution |
| `np.log1p(features)` | Handle extreme skewness (Landsize skew: 40+) |
| `np.expm1(pred)` | Inverse transform to real dollar values |

```python
# Why log1p instead of log?
# log(0) = undefined ❌
# log1p(0) = log(1) = 0 ✅  (handles zero values safely)
```

---

### 🔄 Complete Artifact Serialization

> **Production Pattern:** Serialize entire preprocessing pipeline, not just the model.

**Why this matters:**
- ✅ New predictions use exact same transformations as training
- ✅ No "training-serving skew" in production
- ✅ Model deployable without access to training data

---

## 📁 Project Architecture

```
House-Price-Prediction-Model/
│
├── 📱 app.py                           # Streamlit web application
├── 📋 requirements.txt                 # Python dependencies
│
├── 📊 data/
│   ├── raw/
│   │   └── Melbourne_housing_FULL.csv  # Original dataset (34,857 records)
│   └── processed/                      # Train/test parquet files
│
├── 🤖 models/
│   ├── house_price_model.joblib        # Final production model
│   └── artifacts/                      # 10 preprocessing artifacts
│       ├── target_encoder.joblib       # Suburb/Seller encoder
│       ├── ohe_encoder.joblib          # One-hot encoder
│       ├── scaler_robust.joblib        # RobustScaler
│       ├── scaler_standard.joblib      # StandardScaler
│       ├── suburb_info.joblib          # Suburb → (lat, long, region)
│       └── model_columns.joblib        # Feature column order
│
├── 📓 notebooks/
│   ├── eda.ipynb                       # Exploratory Data Analysis
│   ├── preprocessing.ipynb             # Feature engineering
│   ├── model_training.ipynb            # Model selection & tuning
│   └── evaluation.ipynb                # Performance diagnostics
│
└── 📝 reports/
    └── EDA_notes.md                    # Data insights
```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core ML** | `pandas` `numpy` `scikit-learn` `xgboost` `lightgbm` `category_encoders` |
| **Visualization** | `matplotlib` `seaborn` `plotly` `missingno` |
| **Deployment** | `streamlit` `joblib` `pyarrow` |
| **Development** | `jupyterlab` |

---

## 🚀 Quick Start

### Installation

```bash
# Clone & setup
git clone https://github.com/yourusername/House-Price-Prediction-Model.git
cd House-Price-Prediction-Model

# Create environment
python -m venv env
.\env\Scripts\activate  # Windows
# source env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🧠 Data Science Workflow

### Phase 1: EDA → `notebooks/eda.ipynb`

| Finding | Detail |
|---------|--------|
| **Data Quality** | 34,857 records, 21 features, 1 duplicate |
| **Missing Values** | BuildingArea (50%+), YearBuilt (50%+) |
| **Target** | Right-skewed ($300K - $9M) |
| **Key Correlations** | Rooms↔Price (+0.49), Distance↔Price (-0.32) |

### Phase 2: Preprocessing → `notebooks/preprocessing.ipynb`

```
Raw Data (34,857) → Clean → Impute → Transform → Encode → Clean Data (27,247)
```

### Phase 3: Model Training → `notebooks/model_training.ipynb`

| Model | Performance |
|-------|-------------|
| Linear Regression | Baseline |
| Random Forest | Good |
| **XGBoost** | **Best** |
| LightGBM | Excellent |

### Phase 4: Evaluation → `notebooks/evaluation.ipynb`
- Actual vs Predicted plots
- Residual analysis
- Feature importance ranking

---

## 📈 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~$150,000 | Average prediction error |
| **RMSE** | ~$220,000 | Penalizes large errors |
| **R²** | ~0.75 | Explains 75% of variance |

### Top Features
1. `Suburb` (Target Encoded)
2. `BuildingArea`
3. `Rooms`
4. `Distance` (to CBD)
5. `YearBuilt`

---

## 📚 API Reference

```python
import joblib
import numpy as np

# Load artifacts
model = joblib.load('models/house_price_model.joblib')
suburb_info = joblib.load('models/artifacts/suburb_info.joblib')
# ... load other artifacts

# Create input & preprocess (see app.py for full pipeline)
df = pd.DataFrame([input_data])

# Predict
log_pred = model.predict(df)[0]
price = np.expm1(log_pred)
print(f"Estimated: ${price:,.0f}")
```

---

## 🗂 Dataset Information

**Melbourne Housing Market** - 34,857 historical property sales

| Feature | Type | Notes |
|---------|------|-------|
| `Suburb` | Categorical | 300+ unique values |
| `Rooms` | Numeric | Number of rooms |
| `Type` | Categorical | h=house, u=unit, t=townhouse |
| `Price` | Numeric | **Target variable** |
| `Distance` | Numeric | Distance from CBD (km) |
| `Landsize` | Numeric | Land size (sqm) |
| `BuildingArea` | Numeric | Building size (sqm) |
| `YearBuilt` | Numeric | Construction year |
| `Lattitude/Longtitude` | Numeric | Geographic coordinates |
| `Regionname` | Categorical | General region |

*Dropped: Address, Postcode, CouncilArea, Method, Bedroom2 (redundant/leaky)*

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<div align="center">



[⬆ Back to Top](#-melbourne-house-price-prediction)

</div>
