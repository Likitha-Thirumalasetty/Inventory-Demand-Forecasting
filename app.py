import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Load the saved model
with open('optimized_gbr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the Streamlit app
st.title("Inventory Demand Forecasting")

# Sidebar for user input
st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
age = st.sidebar.selectbox("Age Group", ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
city_category = st.sidebar.selectbox("City Category", ['A', 'B', 'C'])
stay_years = st.sidebar.slider("Years in Current City", 0, 4, 1)
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married'])
product_category = st.sidebar.slider("Product Category", 1, 20, 10)
occupation = st.sidebar.slider("Occupation Code", 0, 20, 5)

# Process input into DataFrame with the correct feature names
input_data = pd.DataFrame({
    'Gender_M': [1 if gender == 'Male' else 0],  # Only Gender_M is used
    'Age_18-25': [1 if age == '18-25' else 0],
    'Age_26-35': [1 if age == '26-35' else 0],
    'Age_36-45': [1 if age == '36-45' else 0],
    'Age_46-50': [1 if age == '46-50' else 0],
    'Age_51-55': [1 if age == '51-55' else 0],
    'Age_55+': [1 if age == '55+' else 0],
    'City_Category_B': [1 if city_category == 'B' else 0],
    'City_Category_C': [1 if city_category == 'C' else 0],
    'Stay_In_Current_City_Years': [stay_years],
    'Marital_Status': [1 if marital_status == 'Married' else 0],
    'Product_Category': [product_category],
    'Occupation': [occupation]
})

# Reorder the columns to match the model's expected order
expected_order = model.feature_names_in_
input_data = input_data[expected_order]

# Predict button
if st.button("Predict Purchase Amount"):
    try:
        # Predict the purchase amount
        prediction = model.predict(input_data)
        st.write(f"### Predicted Purchase Amount: $ {prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")

