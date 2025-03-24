import streamlit as st
import pickle
import numpy as np

# Load models and transformers
with open('fertilizer_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('svd_transform.pkl', 'rb') as f:
    svd_transform = pickle.load(f)

st.title("🌱 Fertilizer Recommendation App")

# Input fields for features
st.write("Enter the following soil and environmental parameters:")
n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, value=50.0)
p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, value=50.0)
k = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, value=50.0)
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)
soil_type = st.selectbox("Soil Type", label_encoders['Soil Type'].classes_)
crop_type = st.selectbox("Crop Type", label_encoders['Crop Type'].classes_)

if st.button("Recommend Fertilizer"):
    # Encode categorical values
    soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
    crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]

    # Create feature vector
    features = np.array([[n, p, k, temp, humidity, moisture, soil_encoded, crop_encoded]])

    # Scale and transform features
    features_scaled = scaler.transform(features)
    features_transformed = svd_transform.transform(features_scaled)

    # Predict
    prediction = model.predict(features_transformed)[0]
    st.success(f"✅ Recommended Fertilizer: {prediction}")

st.caption("Deploy on Streamlit Cloud by pushing to GitHub and linking your repo.")
