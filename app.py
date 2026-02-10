import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Disease Prediction App")

st.write("If the app is working, you should see this text.")

# ----------------------------
# FILE UPLOADER (MANDATORY)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Final_Augmented_dataset_Diseases_and_Symptoms.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆️ Please upload the CSV file to continue")
    st.stop()

# ----------------------------
# LOAD DATA (WITH DEBUG)
# ----------------------------
try:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully")
    st.write("Dataset preview:")
    st.dataframe(data.head())
except Exception as e:
    st.error("❌ Error loading dataset")
    st.write(e)
    st.stop()

# ----------------------------
# SPLIT FEATURES & TARGET
# ----------------------------
try:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
except Exception as e:
    st.error("❌ Error splitting X and y")
    st.write(e)
    st.stop()

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# ----------------------------
# TRAIN MODEL
# ----------------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully")
except Exception as e:
    st.error("❌ Model training failed")
    st.write(e)
    st.stop()

# ----------------------------
# USER INPUT
# ----------------------------
st.subheader("Select Symptoms (1 = Yes, 0 = No)")

user_input = []
cols = st.columns(3)

for i, col in enumerate(X.columns):
    with cols[i % 3]:
        val = st.selectbox(col, [0, 1], key=col)
        user_input.append(val)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🔍 Predict Disease"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        pred = model.predict(input_array)
        disease = le.inverse_transform(pred)

        st.success(f"✅ Predicted Disease: **{disease[0]}**")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)
