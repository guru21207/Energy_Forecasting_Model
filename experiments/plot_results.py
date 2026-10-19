import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

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
df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["hour_weekend"] = df["hour"] * df["is_weekend"]

df = df.dropna().reset_index(drop=True)

features = joblib.load("models/features.pkl")
model = joblib.load("models/final_model.pkl")

split_index = int(len(df) * 0.8)
val_df = df.iloc[split_index:].copy()

X_val = val_df[features]
y_true = val_df["next_interval_Usage_kWh"]

y_pred = np.expm1(model.predict(X_val))
y_pred = np.maximum(y_pred, 0)

plt.figure(figsize=(12, 6))
plt.plot(val_df["date"].iloc[:300], y_true.iloc[:300], label="Actual", linewidth=2)
plt.plot(val_df["date"].iloc[:300], y_pred[:300], label="Predicted", linewidth=2)

plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.show()