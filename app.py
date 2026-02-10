import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# App Title
# ---------------------------
st.set_page_config(page_title="Disease Prediction App")
st.title("🩺 Disease Prediction System")
st.write("Predict disease based on symptoms using Machine Learning")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    return data

data = load_data("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# ---------------------------
# Split Features & Target
# ---------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# ---------------------------
# Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# User Input
# ---------------------------
st.subheader("Select Symptoms (Yes = 1, No = 0)")

user_input = []
for col in X.columns:
    val = st.selectbox(col, [0, 1], key=col)
    user_input.append(val)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Disease"):
    input_data = np.array([user_input])
    prediction = model.predict(input_data)
    disease = le.inverse_transform(prediction)

    st.success(f"🩻 Predicted Disease: **{disease[0]}**")
