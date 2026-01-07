import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

#Page Configuration
st.set_page_config(page_title="Melbourne House Pricer", page_icon="🏠", layout="centered")

@st.cache_resource
def load_system():
    #Define the path to artifacts folder
    ART_PATH = 'models/artifacts/'
    MODEL_PATH = 'models/house_price_model.joblib'
    
    #Load the Model
    model = joblib.load(MODEL_PATH)
    
    #Load the Suburb Dictionary
    suburb_info = joblib.load(f'{ART_PATH}suburb_info.joblib')
    
    #Load Encoders
    te = joblib.load(f'{ART_PATH}target_encoder.joblib')
    ohe = joblib.load(f'{ART_PATH}ohe_encoder.joblib')
    
    #Load Scalers
    s_rob = joblib.load(f'{ART_PATH}scaler_robust.joblib')
    s_std = joblib.load(f'{ART_PATH}scaler_standard.joblib')
    
    #Load Column Maps
    cols_rob = joblib.load(f'{ART_PATH}cols_robust.joblib')
    cols_std = joblib.load(f'{ART_PATH}cols_standard.joblib')
    cols_target = joblib.load(f'{ART_PATH}cols_target.joblib')
    cols_ohe = joblib.load(f'{ART_PATH}cols_ohe.joblib')
    model_cols = joblib.load(f'{ART_PATH}model_columns.joblib')
    
    return model, suburb_info, te, ohe, s_rob, s_std, cols_rob, cols_std, cols_target, cols_ohe, model_cols

try:
    model, suburb_info, te, ohe, s_rob, s_std, cols_rob, cols_std, cols_target, cols_ohe, model_cols = load_system()
    st.toast("System Loaded Successfully!")
except FileNotFoundError as e:
    st.error(f"Critical Error: Missing file. \nDetails: {e}")
    st.stop()

st.title("Melbourne House Value Estimator")
st.markdown("Select a location and property details to get an instant valuation.")


st.subheader("Location")

#Suburb Dropdown
valid_suburbs = sorted(list(suburb_info.keys()))
default_idx = valid_suburbs.index("Richmond") if "Richmond" in valid_suburbs else 0
suburb = st.selectbox("Suburb", valid_suburbs, index=default_idx)

#Retrieve Hidden Data
info = suburb_info[suburb]
auto_lat = info['Lattitude']
auto_long = info['Longtitude']
auto_region = info['Regionname']
auto_prop_count = info['Propertycount']
auto_seller = info['SellerG'] 


st.subheader("Property Specs")

col1, col2 = st.columns(2)
with col1:
    rooms = st.slider("Number of Rooms", 1, 8, 3)
    bathroom = st.slider("Bathrooms", 1, 5, 1)
    car = st.slider("Car Spots", 0, 5, 1)

with col2:
    #Map readable names to model codes
    type_map = {"House": "h", "Unit": "u", "Townhouse": "t"}
    type_display = st.selectbox("Property Type", list(type_map.keys()))
    prop_type = type_map[type_display]
    
    year_built = st.number_input("Year Built", 1850, 2026, 1970)
    distance = st.number_input("Distance to CBD (km)", 0.0, 60.0, 5.0)

    landsize = st.number_input("Land Size (sqm)", value=450)
    building_area = st.number_input("Building Area (sqm)", value=120)

#Prediction Pipeline
if st.button("Estimate Price", type="primary", use_container_width=True):
    
    #Input Validation
    errors = []
    
    if year_built > datetime.datetime.now().year:
        errors.append("❌ Year Built cannot be in the future!")
    
    if landsize <= 0:
        errors.append("❌ Land Size must be greater than 0!")
    
    if building_area <= 0:
        errors.append("❌ Building Area must be greater than 0!")
    
    #Display errors and stop if any validation fails
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    
    #Create the Raw Input DataFrame
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
        'Regionname': auto_region,      
        'Propertycount': auto_prop_count, 
        'Lattitude': auto_lat,          
        'Longtitude': auto_long,        
        'SoldYear': datetime.datetime.now().year
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        #Preprocessing Steps

        #Log Transform
        skewed_cols = ['Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount']
        for col in skewed_cols:
            df[col] = np.log1p(df[col])

        #Scaling
        #Apply RobustScaler
        if len(cols_rob) > 0:
            df[cols_rob] = s_rob.transform(df[cols_rob])
        #Apply StandardScaler
        if len(cols_std) > 0:
            df[cols_std] = s_std.transform(df[cols_std])

        #One-Hot Encoding
        ohe_df = ohe.transform(df[cols_ohe])
        df = pd.concat([df, ohe_df], axis=1).drop(columns=cols_ohe)

        #Target Encoding
        df[cols_target] = te.transform(df[cols_target])

        #Ensures exact column match with X_train
        df = df.reindex(columns=model_cols, fill_value=0)
        
        #Predict
        log_pred = model.predict(df)[0]
        
        #Inverse Log Transform
        price = np.expm1(log_pred)
        
        #Prediction Confidence Interval (Console Log)
        print("\nPREDICTION RESULT")
        print(f"Estimated Price: ${price:,.0f}")
        
        if hasattr(model, 'estimators_'):  
            std = np.std([np.expm1(tree.predict(df)[0]) for tree in model.estimators_])
            cv = (std / price) * 100  #Coefficient of variation
            confidence_pct = max(0, min(100, 100 - cv))  #Convert to confidence percentage
            print(f"Confidence Range: ${price-std:,.0f} - ${price+std:,.0f}")
            print(f"Standard Deviation: ${std:,.0f}")
            print(f"Prediction Confidence: {confidence_pct:.1f}%")
        elif hasattr(model, 'booster_'):  
            #Get predictions from each tree iteration
            n_iterations = model.n_estimators
            tree_preds = []
            for i in range(1, n_iterations + 1, max(1, n_iterations // 20)):  # Sample 20 checkpoints
                pred = model.predict(df, num_iteration=i)[0]
                tree_preds.append(np.expm1(pred))
            std = np.std(tree_preds) * 2  #Scale up since sampling progression
            cv = (std / price) * 100  #Coefficient of variation
            confidence_pct = max(0, min(100, 100 - cv))  #Convert to confidence percentage
            print(f"Confidence Range: ${price-std:,.0f} - ${price+std:,.0f}")
            print(f"Estimated Uncertainty: ${std:,.0f}")
            print(f"Prediction Confidence: {confidence_pct:.1f}%")
            print(f"Model Iterations: {n_iterations}")
        else:
            print("Note: Confidence interval not available for this model type")
    
        #Display Result
        st.success(f"Estimated Value: ${price:,.0f}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")