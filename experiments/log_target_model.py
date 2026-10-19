import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
split = len(df) - 5607

train = df.iloc[:split]
valid = df.iloc[split:]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

def make_model():
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=42
        ))
    ])

baseline = make_model()
baseline.fit(train[features], train[target])
baseline_pred = baseline.predict(valid[features])

log_model = make_model()
log_model.fit(train[features], np.log1p(train[target]))
log_pred = np.expm1(log_model.predict(valid[features]))
log_pred = np.maximum(log_pred, 0)

for name, predictions in [
    ("Existing model", baseline_pred),
    ("Log-target model", log_pred)
]:
    mae = mean_absolute_error(valid[target], predictions)
    rmse = np.sqrt(mean_squared_error(valid[target], predictions))

    print(f"\n{name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

print("\nValidation period:")
print(valid["date"].min())
print(valid["date"].max())

print("\n========== LOG MODEL PERFORMANCE BY LOAD TYPE ==========")

results = valid[["Load_Type", target]].copy()
results["predicted"] = log_pred
results["absolute_error"] = (
    results[target] - results["predicted"]
).abs()

print(results.groupby("Load_Type").agg(
    Rows=("absolute_error", "size"),
    MAE=("absolute_error", "mean"),
    Maximum_Error=("absolute_error", "max")
).round(4))