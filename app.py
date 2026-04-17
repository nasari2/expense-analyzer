import streamlit as st
import pandas as pd
import joblib

st.title("💰 Personal Expense Analyzer")

file = st.file_uploader("Upload your expense CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.write(df)

    df['Date'] = pd.to_datetime(df['Date'])

    st.subheader("💵 Total Expense")
    st.write(df['Amount'].sum())

    st.subheader("📂 Category-wise Spending")
    category = df.groupby('Category')['Amount'].sum()
    st.bar_chart(category)

    model = joblib.load("model.pkl")

    st.subheader("🔮 Predict Future Expense")
    day = st.slider("Select Day", 1, 31)
    pred = model.predict([[day]])

    st.write(f"Estimated Expense: ₹{pred[0]:.2f}")

    st.subheader("⚠️ Unusual Spending")
    threshold = df['Amount'].mean() * 1.5
    anomalies = df[df['Amount'] > threshold]

    if anomalies.empty:
        st.write("No unusual spending detected")
    else:
        st.write(anomalies)