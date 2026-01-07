# 🏠 Melbourne House Price Prediction Model

An end-to-end machine learning solution designed to estimate residential property prices in Melbourne, Australia. This project demonstrates a complete data science lifecycle, from exploratory data analysis and advanced preprocessing to model training, hyperparameter tuning, and deployment via an interactive Streamlit web application.

## 📌 Project Overview

Real estate valuation is a complex problem influenced by numerous location and property-specific factors. This project leverages historical market data to build a predictive model that provides instant valuations.

**Key Highlights:**
*   **Data Pipeline:** A robust cleaning and feature engineering process that handles missing values (KNN & Centroid imputation) and skewed distributions (`log1p` transforms).
*   **Model Optimization:** Benchmarks multiple algorithms (Linear Regression, Random Forest, XGBoost, LightGBM) with `RandomizedSearchCV` for hyperparameter tuning.
*   **Production-Ready:** Includes a persisted preprocessing pipeline (Scalers/Encoders) to ensure new data is treated exactly like training data.
*   **Interactive UI:** A deployed Streamlit application (`app.py`) for easy user interaction and real-time inference.

## 📂 Repository Structure

```text
House-Price-Prediction-Model/
├── app.py                      # Main Streamlit application entry point
├── models/                     # Model artifacts and serialized objects
│   ├── house_price_model.joblib # The final production model
│   └── artifacts/              # Preprocessing artifacts (Encoders, Scalers)
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── eda.ipynb               # Exploratory Data Analysis & insights
│   ├── preprocessing.ipynb     # Feature engineering and cleaning logic
│   ├── model_training.ipynb    # Model training, comparison, and tuning
│   └── evaluation.ipynb        # Performance metrics and residual analysis
├── data/                       # Dataset storage
│   ├── raw/                    # Raw CSV data
│   └── processed/              # Processed datasets for training
├── reports/                    # Documentation
│   └── EDA_notes.md            # Detailed notes on data findings
└── requirements.txt            # Project dependencies
```

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
*   Python 3.10+ recommended
*   pip package manager

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd House-Price-Prediction-Model
    ```

2.  **Set Up Virtual Environment**
    ```bash
    python -m venv env
    # Windows
    .\env\Scripts\activate
    # Mac/Linux
    source env/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the value estimator dashboard:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🧠 Data Science Workflow

### 1. Exploratory Data Analysis (EDA)
Located in `notebooks/eda.ipynb`. Key steps included:
*   Identifying "Date" effects (Inflation) and extracting `Sold_Year`.
*   Analyzing correlations between `Rooms`, `Landsize`, and `Price`.
*   Detecting errors (e.g., YearBuilt > 2024).

### 2. Preprocessing & Feature Engineering
Located in `notebooks/preprocessing.ipynb`. Strategies applied:
*   **Inconsistency Handling:** Correcting Suburb names and Seller info.
*   **Imputation:** 
    *   *KNN Imputation*: For `Landsize`, `BuildingArea`, `YearBuilt`.
    *   *Centroid Imputation*: For missing `Latitude`/`Longitude` based on Suburb.
*   **Transformation:** Log-transformation of target `Price` and skewed features like `Distance`.
*   **Encoding:** One-Hot Encoding for categorical regions; Target Encoding for high-cardinality suburbs.

### 3. Model Training & Evaluation
Located in `notebooks/model_training.ipynb`.
*   **Algorithms Tested:** Linear Regression, Random Forest, XGBoost, LightGBM.
*   **Selection:** The best performing model (lowest MAE) was selected after Hyperparameter Tuning.
*   **Evaluation:** performance measured using Mean Absolute Error (MAE) and R-squared ($R^2$).

## 🛠️ Technologies Used

*   **Core:** Python, Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM
*   **Deployment:** Streamlit, Joblib
*   **Visualization:** Plotly, Seaborn, Matplotlib

## 📝 License

This project is open-source and available under the standard MIT License.

