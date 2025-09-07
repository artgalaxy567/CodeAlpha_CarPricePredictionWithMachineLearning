# Car Price Prediction with Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("car data.csv")

# 2. Data Preprocessing
df = df.drop(columns=["Car_Name"])  # drop Car_Name (not useful for regression)

# Encode categorical variables
label_enc = LabelEncoder()
df["Fuel_Type"] = label_enc.fit_transform(df["Fuel_Type"])
df["Selling_type"] = label_enc.fit_transform(df["Selling_type"])
df["Transmission"] = label_enc.fit_transform(df["Transmission"])

# Features (X) and Target (y)
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# 6. Visualization: Actual vs Predicted Prices
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # reference line
plt.show()
