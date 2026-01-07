## Melbourne House Price Prediction Model

Interactive Streamlit application and end‑to‑end ML pipeline for estimating residential property values in Melbourne, based on the `Melbourne_housing_FULL.csv` dataset. The project covers data exploration, preprocessing, model training, evaluation, and deployment of a production‑style prediction service.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data & Preprocessing](#data--preprocessing)
- [Model](#model)
- [Running the App](#running-the-app)
- [Industrial Practices & Design Choices](#industrial-practices--design-choices)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project builds a house price prediction model for Melbourne and exposes it via a user‑friendly Streamlit web app. Users select a suburb and specify property characteristics (rooms, bathrooms, car spaces, building area, land size, year built, etc.), and the app returns an estimated sale price along with a confidence indicator.

The full workflow includes:

- Exploratory Data Analysis (EDA) and documentation of decisions.
- Systematic data cleaning, feature engineering, and encoding.
- Training a tree‑based ensemble regressor and persisting all preprocessing artifacts.
- Deploying the trained model as an interactive application.

---

## Features

- **Interactive valuation UI** built with Streamlit (`app.py`).
- **Suburb‑aware predictions** using suburb‑level metadata (latitude, longitude, region, property count, seller information).
- **Robust preprocessing pipeline** with log transforms, robust scaling, one‑hot encoding, and target encoding.
- **Consistent train/inference behavior** via saved encoders, scalers, and column lists.
- **Basic uncertainty estimation**, showing a confidence range derived from the underlying ensemble.
- **Clearly documented EDA & preprocessing decisions** in markdown and notebooks.

---

## Project Structure

At a high level, the repository is organized as follows:

- `app.py` – Streamlit application that loads artifacts, validates inputs, runs the preprocessing pipeline, and serves predictions.
- `requirements.txt` – Python dependencies for the environment.
- `data/`
	- `raw/`
		- `Melbourne_housing_FULL.csv` – Original dataset.
	- `processed/` – Placeholder for cleaned/feature‑engineered datasets.
- `models/`
	- `house_price_model.joblib` – Trained regression model.
	- `artifacts/`
		- `suburb_info.joblib` – Dictionary of suburb‑level metadata used in the UI and model.
		- `target_encoder.joblib` – Target encoder for high‑cardinality categorical features.
		- `ohe_encoder.joblib` – One‑hot encoder for low‑cardinality categorical features.
		- `scaler_robust.joblib`, `scaler_standard.joblib` – Fitted scalers for numeric features.
		- `cols_robust.joblib`, `cols_standard.joblib`, `cols_target.joblib`, `cols_ohe.joblib` – Definitions of which columns each transformer applies to.
		- `model_columns.joblib` – Final feature column order expected by the trained model.
- `notebooks/`
	- `eda.ipynb` – Exploratory data analysis.
	- `preprocessing.ipynb` – Data cleaning and feature engineering.
	- `model_training.ipynb` – Model training workflow.
	- `evaluation.ipynb` – Model evaluation and diagnostics.
- `reports/`
	- `EDA_notes.md` – Narrative description of EDA findings and preprocessing strategy.
- `src/` – Reserved for reusable Python modules (future extension of production code).
- `env/` – Local virtual environment (not required if you create your own).

---

## Data & Preprocessing

The preprocessing design is documented in detail in `reports/EDA_notes.md`. Key steps include:

### 1. Data Cleaning

- Convert the `Date` column to a proper datetime type and extract `Sold_Year`.
- Remove duplicate rows.
- Handle logical inconsistencies, e.g.:
	- `BuildingArea = 0` → treated as missing.
	- `YearBuilt` constrained to a valid range (1800–current year); out‑of‑range values → missing.
	- `Bathrooms = 0` treated as likely data error except for properties before 1900.
- Normalize text and handle near‑duplicates:
	- Standardize `Suburb` casing.
	- Normalize `SellerG` (e.g., stripping suffixes after `/` and lowercasing).

### 2. Feature Selection

Drop columns that behave like identifiers or add little generalizable signal, such as:

- `Address` (high cardinality, acts as ID).
- `Bedroom2` (redundant with `Rooms`).
- `CouncilArea`, `Postcode` when redundant with other spatial features.
- `Method` due to potential leakage (describes sale outcome).
- Rows with missing target (`Price`).

### 3. Missing Value Imputation

Different strategies are used based on variable type and missingness pattern:

- **Row drops** for critical fields: remove rows with missing `Price` or `Postcode`.
- **KNN imputation** for structural missingness in numeric features:
	- `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Bathroom`.
- **Suburb centroid imputation** for spatial fields:
	- `Lattitude`, `Longtitude` (using suburb‑level aggregates, after train/test split).
- **Simple statistical imputation** (mean/median/mode) for:
	- `Regionname`, `Propertycount`, `Distance`.

### 4. Feature Engineering & Transformations

- Create `SoldYear` from the sale date to capture temporal effects (e.g. inflation).
- Group rare region categories into `Regional Victoria`.
- Use `np.log1p` to reduce skew for:
	- Target: `Price` (during training).
	- Predictors: `Distance`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `Propertycount`.
- Leave reasonably symmetric or already bounded features (e.g. `Rooms`, `Lattitude`, `Longtitude`) untransformed.

### 5. Scaling & Encoding

- **RobustScaler** for outlier‑sensitive numeric features (`Rooms`, `Distance`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `Propertycount`).
- **StandardScaler** for more stable numeric features (`YearBuilt`, `Lattitude`, `Longtitude`).
- **One‑hot encoding** for low‑cardinality categoricals (`Type`, grouped `Regionname`).
- **Target encoding** for high‑cardinality categoricals (`Suburb`, `SellerG`).

All fitted transformers and the column subsets they operate on are saved into `models/artifacts/` and re‑used at prediction time for strict train/inference consistency.

---

## Model

The trained model is a tree‑based ensemble regressor serialized as `models/house_price_model.joblib`. While the exact estimator type is abstracted behind the joblib file, the prediction interface follows a standard scikit‑learn‑style `predict` API.

In `app.py`, the following steps are performed at inference time:

1. Construct a single‑row `pandas.DataFrame` from user inputs.
2. Apply the same log transforms, scalers (robust/standard), one‑hot encoder, and target encoder as used during training.
3. Reindex the feature matrix to match `model_columns` exactly, ensuring compatibility with the trained model.
4. Obtain a **log‑space prediction** and inverse‑transform it with `np.expm1` to get a price in dollars.
5. Optionally approximate a **confidence range**:
	 - For ensemble models exposing individual estimators, compute dispersion across trees.
	 - For iterative boosting models, sample predictions across iterations.

---

## Running the App

### 1. Set Up Environment

It is recommended to use a virtual environment (you can reuse the existing `env/` or create a new one).

```bash
python -m venv env
env\Scripts\activate  # On Windows

pip install -r requirements.txt
```

### 2. Start the Streamlit App

From the project root:

```bash
streamlit run app.py
```

This will start a local server and open the **Melbourne House Value Estimator** in your browser.

### 3. Using the App

1. Select a **suburb** from the dropdown.
2. Adjust property **rooms**, **bathrooms**, **car spaces**, **land size**, **building area**, **year built**, and **distance to CBD**.
3. Click **"Estimate Price"**.
4. If inputs are valid, the app displays an estimated value; otherwise, detailed validation errors are shown.

---

## Industrial Practices & Design Choices

This project incorporates several practices aligned with real‑world, production‑grade ML systems:

### 1. Reproducible Environments & Dependencies

- Use of a dedicated virtual environment (`env/`).
- Explicit `requirements.txt` with pinned versions for core libraries (pandas, numpy, scikit‑learn, xgboost, lightgbm, streamlit, etc.).
- Clear separation between environment setup and project code.

### 2. Clear Separation of Concerns

- **Experimentation vs. production**:
	- Jupyter notebooks in `notebooks/` for EDA, preprocessing design, model training, and evaluation.
	- A focused application entry point (`app.py`) for serving predictions.
- **Data lifecycle separation**:
	- Distinct `data/raw/` and `data/processed/` folders.
- **Model vs. artifacts**:
	- Model and all preprocessing assets stored under `models/` and `models/artifacts/` for easy deployment and versioning.
- **Documentation**: Narrative EDA and preprocessing rationale in `reports/EDA_notes.md`.

### 3. Production‑Style Preprocessing Pipeline

- Every transformation used in training (log transforms, scaling, encodings) is serialized and re‑applied at inference.
- Column subsets for each transformer are saved (`cols_robust`, `cols_standard`, `cols_target`, `cols_ohe`) to avoid hard‑coding feature indices.
- Final model input schema (`model_columns`) is stored and enforced via `DataFrame.reindex`, reducing risk of column mismatch when models or features change.
- Systematic treatment of missing data with appropriate strategies (KNN imputation, centroid imputation, statistical imputation) driven by EDA insights.

### 4. Robust Inference Service

- **Caching of heavy resources** with `@st.cache_resource` to avoid reloading large models and artifacts on each interaction.
- **Defensive loading** with explicit error handling for missing files (`FileNotFoundError`) and graceful app termination when critical artifacts are unavailable.
- **Input validation** before prediction:
	- Prevents impossible values (e.g. future `Year Built`, non‑positive land/building size).
	- Collects and presents user‑friendly error messages.
- **Basic model‑uncertainty signal** by inspecting variation across ensemble components or iterations, translated into a confidence percentage and range.

### 5. MLOps‑Friendly Design

- All model‑time dependencies (model, encoders, scalers, column lists, metadata) are bundled as artifacts, enabling straightforward redeployment without notebooks.
- The app code (`app.py`) is thin and mostly orchestrates loading artifacts, validating inputs, applying the pipeline, and calling `predict`.
- The repository layout (`data/`, `models/`, `notebooks/`, `reports/`, `src/`) matches common patterns in production ML projects, easing collaboration and future automation (CI/CD, retraining pipelines, etc.).

---

## Future Improvements

- Add automated tests for the preprocessing pipeline and prediction logic.
- Move reusable preprocessing code from notebooks into importable modules under `src/`.
- Implement model versioning and experiment tracking (e.g., MLflow or similar).
- Add richer explainability in the app (feature importance, SHAP values, or partial dependence plots).
- Containerize the application (Docker) and integrate into a CI/CD pipeline for deployment.
