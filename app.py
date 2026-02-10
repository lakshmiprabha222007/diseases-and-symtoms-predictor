import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Disease Prediction App", layout="wide")

st.title("🩺 Disease Prediction System")
st.write("Machine Learning based disease prediction using symptoms")

# --------------------------------------------------
# Load Dataset (CORRECTED)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

data = load_data()

# --------------------------------------------------
# Separate Features & Target
# --------------------------------------------------
X = data.iloc[:, :-1]   # Symptoms
y = data.iloc[:, -1]    # Disease

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Convert all symptom values to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train Random Forest Model
# --------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.subheader("Select Symptoms (1 = Yes, 0 = No)")

user_input = []
cols = st.columns(3)

for i, symptom in enumerate(X.columns):
    with cols[i % 3]:
        value = st.selectbox(symptom, [0, 1], key=symptom)
        user_input.append(value)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Disease"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    disease_name = le.inverse_transform(prediction)

    st.success(f"✅ Predicted Disease: **{disease_name[0]}**")
