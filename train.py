import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("sample_data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day

X = df[['Day']]
y = df['Amount']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully!")