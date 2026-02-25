import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Food Delivery Time Predictor", layout="wide")

st.title("🚚 Food Delivery Time Prediction App")
st.write("Enter order details below to predict delivery time.")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Food.csv")
    return df

df = load_data()

# -----------------------------
# Data Cleaning
# -----------------------------
df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
df["Traffic_Level"] = df["Traffic_Level"].fillna(df["Traffic_Level"].mode()[0])
df["Time_of_Day"] = df["Time_of_Day"].fillna(df["Time_of_Day"].mode()[0])
df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].mode()[0])

# -----------------------------
# Feature Selection
# -----------------------------
x_num = df[["Distance_km", "Preparation_Time_min"]]
y = df["Delivery_Time_min"]

features_cat = df[["Weather", "Traffic_Level", "Time_of_Day", 
                   "Vehicle_Type", "Courier_Experience_yrs"]]

# One Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(features_cat)
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(features_cat.columns))

# Final X
x_final = pd.concat([x_num, encoded_df], axis=1)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_final, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# Metrics
test_score = model.score(x_test, y_test)
y_pred_test = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred_test)

# -----------------------------
# Sidebar Model Info
# -----------------------------
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**R² Score:** {round(test_score,3)}")
st.sidebar.write(f"**Mean Absolute Error:** {round(mae,2)} minutes")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("📝 Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
    prep_time = st.number_input("Preparation Time (minutes)", min_value=0, step=1)
    weather = st.selectbox("Weather", df["Weather"].unique())
    traffic = st.selectbox("Traffic Level", df["Traffic_Level"].unique())

with col2:
    time_day = st.selectbox("Time of Day", df["Time_of_Day"].unique())
    vehicle = st.selectbox("Vehicle Type", df["Vehicle_Type"].unique())
    experience = st.selectbox("Courier Experience (yrs)", 
                              df["Courier_Experience_yrs"].unique())

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Delivery Time"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Preparation_Time_min": [prep_time],
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Time_of_Day": [time_day],
        "Vehicle_Type": [vehicle],
        "Courier_Experience_yrs": [experience]
    })

    # Encode categorical
    input_cat = input_data[["Weather","Traffic_Level",
                            "Time_of_Day","Vehicle_Type",
                            "Courier_Experience_yrs"]]

    input_encoded = ohe.transform(input_cat)
    input_encoded_df = pd.DataFrame(
        input_encoded,
        columns=ohe.get_feature_names_out(input_cat.columns)
    )

    input_final = pd.concat(
        [input_data[["Distance_km","Preparation_Time_min"]],
         input_encoded_df],
        axis=1
    )

    # Match columns
    input_final = input_final.reindex(columns=x_final.columns, fill_value=0)

    prediction = model.predict(input_final)

    st.success(f"🚀 Estimated Delivery Time: {round(prediction[0],2)} minutes")