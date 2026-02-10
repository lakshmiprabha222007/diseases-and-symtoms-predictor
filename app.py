import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Disease Prediction", layout="wide")

st.title("🩺 Disease Prediction App")
st.write("Upload the dataset and predict disease based on symptoms")

# -----------------------------
# File uploader (NO st.stop)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Final_Augmented_dataset_Diseases_and_Symptoms.csv",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully")
    st.dataframe(data.head())

    # -------------------------
    # Split features & target
    # -------------------------
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # -------------------------
    # Train model
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model trained successfully")

    # -------------------------
    # User input
    # -------------------------
    st.subheader("Select Symptoms (1 = Yes, 0 = No)")

    user_input = []
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            val = st.selectbox(col, [0, 1], key=col)
            user_input.append(val)

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("🔍 Predict Disease"):
        input_array = np.array(user_input).reshape(1, -1)
        pred = model.predict(input_array)
        disease = le.inverse_transform(pred)

        st.success(f"✅ Predicted Disease: **{disease[0]}**")

else:
    st.info("⬆️ Please upload the CSV file to continue")
