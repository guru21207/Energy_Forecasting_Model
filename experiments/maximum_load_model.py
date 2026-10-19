import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y %H:%M"
)

df = df.sort_values("date").reset_index(drop=True)

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

usage_by_time = df.set_index("date")["Usage_kWh"]

for lag, minutes in [(1, 15), (4, 60), (96, 1440)]:
    previous_time = df["date"] - pd.Timedelta(minutes=minutes)
    df[f"usage_lag_{lag}"] = previous_time.map(usage_by_time)

features = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
    "hour",
    "minute",
    "day",
    "month",
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "hour_weekend",
    "usage_lag_1",
    "usage_lag_4",
    "usage_lag_96"
]

target = "next_interval_Usage_kWh"
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
val = df.iloc[split_index:].copy()

categorical_features = ["WeekStatus", "Day_of_week", "Load_Type"]
numerical_features = [
    col for col in features if col not in categorical_features
]

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numerical_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=42
    ))
])

baseline.fit(train[features], train[target])
baseline_predictions = baseline.predict(val[features])

specialized_train = train[train["Load_Type"] == "Maximum_Load"]
specialized_val_mask = val["Load_Type"] == "Maximum_Load"

specialized_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=42
    ))
])

specialized_model.fit(
    specialized_train[features],
    specialized_train[target]
)

specialized_predictions = baseline_predictions.copy()

specialized_predictions[specialized_val_mask.to_numpy()] = (
    specialized_model.predict(val.loc[specialized_val_mask, features])
)

y_val = val[target].to_numpy()

baseline_max = baseline_predictions[specialized_val_mask.to_numpy()]
specialized_max = specialized_predictions[specialized_val_mask.to_numpy()]
actual_max = y_val[specialized_val_mask.to_numpy()]

print("\n========== MAXIMUM LOAD MODEL COMPARISON ==========")

print("\nExisting model — Maximum_Load:")
print(f"Rows: {len(actual_max)}")
print(f"MAE: {mean_absolute_error(actual_max, baseline_max):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual_max, baseline_max)):.4f}")

print("\nSpecialized model — Maximum_Load:")
print(f"Rows: {len(actual_max)}")
print(f"MAE: {mean_absolute_error(actual_max, specialized_max):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual_max, specialized_max)):.4f}")

print("\n========== OVERALL VALIDATION COMPARISON ==========")

print("\nExisting model:")
print(f"MAE: {mean_absolute_error(y_val, baseline_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, baseline_predictions)):.4f}")

print("\nCombined model:")
print(f"MAE: {mean_absolute_error(y_val, specialized_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, specialized_predictions)):.4f}")

print("\nValidation period:")
print(val["date"].iloc[0])
print(val["date"].iloc[-1])
