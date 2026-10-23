import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)


# ---------------------------------------------------
# 1. Generate a sample rainfall dataset
# ---------------------------------------------------

np.random.seed(42)

rows = []

for year in range(2015, 2026):
    for month in range(1, 13):

        # Hyderabad-like seasonal pattern
        if month in [6, 7, 8, 9]:
            base_rainfall = np.random.uniform(100, 250)
            humidity = np.random.uniform(65, 90)
            temperature = np.random.uniform(25, 32)
        elif month in [10, 11]:
            base_rainfall = np.random.uniform(50, 150)
            humidity = np.random.uniform(55, 80)
            temperature = np.random.uniform(24, 31)
        else:
            base_rainfall = np.random.uniform(0, 40)
            humidity = np.random.uniform(35, 65)
            temperature = np.random.uniform(28, 38)

        wind_speed = np.random.uniform(2, 15)
        previous_rainfall = np.random.uniform(0, 200)

        rainfall = (
            base_rainfall
            + (humidity - 60) * 2
            - (temperature - 30) * 1.5
            + previous_rainfall * 0.15
            + np.random.normal(0, 15)
        )

        rainfall = max(0, rainfall)

        rows.append([
            year,
            month,
            round(temperature, 2),
            round(humidity, 2),
            round(wind_speed, 2),
            round(previous_rainfall, 2),
            round(rainfall, 2)
        ])


df = pd.DataFrame(rows, columns=[
    "Year",
    "Month",
    "Temperature",
    "Humidity",
    "WindSpeed",
    "PreviousRainfall",
    "Rainfall"
])


# Save dataset
df.to_csv("data/rainfall.csv", index=False)

print("Dataset created successfully!")
print(f"Total records: {len(df)}")


# ---------------------------------------------------
# 2. Prepare data for Machine Learning
# ---------------------------------------------------

X = df[
    [
        "Year",
        "Month",
        "Temperature",
        "Humidity",
        "WindSpeed",
        "PreviousRainfall"
    ]
]

y = df["Rainfall"]


# ---------------------------------------------------
# 3. Split dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# 4. Train Random Forest Regression model
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# ---------------------------------------------------
# 6. Save trained model
# ---------------------------------------------------

with open("model/rainfall_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
print("Location: model/rainfall_model.pkl")