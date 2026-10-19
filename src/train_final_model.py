import pandas as pd
import numpy as np
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
df = df.sort_values("date").reset_index(drop=True)

usage_by_time = df.set_index("date")["Usage_kWh"]

for lag, minutes in [(1, 15), (4, 60), (96, 1440)]:
    previous_time = df["date"] - pd.Timedelta(minutes=minutes)
    df[f"usage_lag_{lag}"] = previous_time.map(usage_by_time)

df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["hour_weekend"] = df["hour"] * df["is_weekend"]

df = df.dropna(subset=["usage_lag_1", "usage_lag_4", "usage_lag_96"])

target = "next_interval_Usage_kWh"

categorical = ["WeekStatus", "Day_of_week", "Load_Type"]

numerical = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "hour", "minute", "day", "month", "day_of_year",
    "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "hour_weekend",
    "usage_lag_1", "usage_lag_4", "usage_lag_96"
]

features = numerical + categorical

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=42
    ))
])

model.fit(df[features], np.log1p(df[target]))

joblib.dump(model, "models/final_model.pkl")
joblib.dump(features, "models/features.pkl")

print("Final model trained successfully.")
print("Model saved to models/final_model.pkl")
print("Features saved to models/features.pkl")
print(f"Training rows: {len(df)}")
print("Target transformation: log1p")