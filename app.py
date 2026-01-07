import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Melbourne House Pricer", page_icon="🏠", layout="centered")

# --- 1. LOAD THE SYSTEM ---
@st.cache_resource
def load_system():
    # Define the path to your artifacts folder
    ART_PATH = 'models/artifacts/'
    MODEL_PATH = 'models/house_price_model.joblib'
    
    # A. Load the Model
    model = joblib.load(MODEL_PATH)
    
    # B. Load the Suburb Brain (The Dictionary)
    # Contains Lat, Long, Region, Density, AND the Dominant Seller
    suburb_info = joblib.load(f'{ART_PATH}suburb_info.joblib')
    
    # C. Load Encoders
    te = joblib.load(f'{ART_PATH}target_encoder.joblib')
    ohe = joblib.load(f'{ART_PATH}ohe_encoder.joblib')
    
    # D. Load Scalers
    s_rob = joblib.load(f'{ART_PATH}scaler_robust.joblib')
    s_std = joblib.load(f'{ART_PATH}scaler_standard.joblib')
    
    # E. Load Column Maps
    cols_rob = joblib.load(f'{ART_PATH}cols_robust.joblib')
    cols_std = joblib.load(f'{ART_PATH}cols_standard.joblib')
    cols_target = joblib.load(f'{ART_PATH}cols_target.joblib')
    cols_ohe = joblib.load(f'{ART_PATH}cols_ohe.joblib')
    model_cols = joblib.load(f'{ART_PATH}model_columns.joblib')
    
    return model, suburb_info, te, ohe, s_rob, s_std, cols_rob, cols_std, cols_target, cols_ohe, model_cols

try:
    model, suburb_info, te, ohe, s_rob, s_std, cols_rob, cols_std, cols_target, cols_ohe, model_cols = load_system()
    st.toast("System Loaded Successfully!", icon="✅")
except FileNotFoundError as e:
    st.error(f"❌ Critical Error: Missing file. \nDetails: {e}")
    st.stop()


# --- 2. USER INTERFACE ---
st.title("🏡 Melbourne House Value Estimator")
st.markdown("Select a location and property details to get an instant valuation.")

# --- SECTION A: LOCATION (Smart Lookup) ---
st.subheader("📍 Location")

# 1. Suburb Dropdown
valid_suburbs = sorted(list(suburb_info.keys()))
default_idx = valid_suburbs.index("Richmond") if "Richmond" in valid_suburbs else 0
suburb = st.selectbox("Suburb", valid_suburbs, index=default_idx)

# 2. Retrieve Hidden Data (The "Brain" part)
info = suburb_info[suburb]
auto_lat = info['Lattitude']
auto_long = info['Longtitude']
auto_region = info['Regionname']
auto_prop_count = info['Propertycount']
auto_seller = info['SellerG'] # <--- Automatically gets the top agent for this suburb

# --- SECTION B: HOUSE DETAILS ---
st.subheader("🏠 Property Specs")

col1, col2 = st.columns(2)
with col1:
    rooms = st.slider("Number of Rooms", 1, 8, 3)
    bathroom = st.slider("Bathrooms", 1, 5, 1)
    car = st.slider("Car Spots", 0, 5, 1)

with col2:
    # Map readable names to model codes
    type_map = {"House": "h", "Unit": "u", "Townhouse": "t"}
    type_display = st.selectbox("Property Type", list(type_map.keys()))
    prop_type = type_map[type_display]
    
    year_built = st.number_input("Year Built", 1850, 2026, 1970)
    distance = st.number_input("Distance to CBD (km)", 0.0, 60.0, 5.0)

    landsize = st.number_input("Land Size (sqm)", value=450)
    building_area = st.number_input("Building Area (sqm)", value=120)

# --- 3. PREDICTION PIPELINE ---
if st.button("🚀 Estimate Price", type="primary", use_container_width=True):
    
    # Input Validation
    errors = []
    
    if year_built > datetime.datetime.now().year:
        errors.append("❌ Year Built cannot be in the future!")
    
    if landsize <= 0:
        errors.append("❌ Land Size must be greater than 0!")
    
    if building_area <= 0:
        errors.append("❌ Building Area must be greater than 0!")
    
    # Display errors and stop if any validation fails
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    
    # A. Create the Raw Input DataFrame
    input_data = {
        'Suburb': suburb,
        'Rooms': rooms,
        'Type': prop_type,   
        'SellerG': auto_seller,         
        'Distance': distance,             
        'Bathroom': bathroom,
        'Car': car,
        'Landsize': landsize,
        'BuildingArea': building_area,
        'YearBuilt': year_built,
        'Regionname': auto_region,      # Inferred
        'Propertycount': auto_prop_count, # Inferred
        'Lattitude': auto_lat,          # Inferred
        'Longtitude': auto_long,        # Inferred
        'SoldYear': datetime.datetime.now().year
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        # --- REPLICATING PIPELINE (CORRECT NOTEBOOK ORDER) ---
        
        # 1. Log Transform
        # Applied to skewed features first
        skewed_cols = ['Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount']
        for col in skewed_cols:
            df[col] = np.log1p(df[col])

        # 2. Scaling (MUST happen BEFORE Encoding)
        # Apply RobustScaler
        if len(cols_rob) > 0:
            df[cols_rob] = s_rob.transform(df[cols_rob])
        # Apply StandardScaler (Includes 'SoldYear' if it was in the training list)
        if len(cols_std) > 0:
            df[cols_std] = s_std.transform(df[cols_std])

        # 3. One-Hot Encoding
        ohe_df = ohe.transform(df[cols_ohe])
        df = pd.concat([df, ohe_df], axis=1).drop(columns=cols_ohe)

        # 4. Target Encoding
        df[cols_target] = te.transform(df[cols_target])

        # 5. Alignment
        # Ensures exact column match with X_train (Handle 'Method' or missing cols)
        df = df.reindex(columns=model_cols, fill_value=0)
        
        # 6. Predict
        log_pred = model.predict(df)[0]
        
        # 7. Inverse Log Transform
        price = np.expm1(log_pred)
        
        # --- DISPLAY RESULT ---
        st.success(f"💰 Estimated Value: ${price:,.0f}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")