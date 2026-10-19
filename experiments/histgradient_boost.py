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

for lag, minutes in [
    (1, 15),
    (4, 60),
    (96, 1440)
]:
    previous_time = df["date"] - pd.Timedelta(minutes=minutes)
    df[f"usage_lag_{lag}"] = previous_time.map(usage_by_time)

target = "next_interval_Usage_kWh"

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

X = df[features]
y = df[target]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

categorical_features = [
    "WeekStatus",
    "Day_of_week",
    "Load_Type"
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)

configs = [
    (250, 0.05, 31, 0.0),
    (200, 0.03, 31, 1.0),
    (150, 0.10, 31, 0.0),
    (300, 0.05, 20, 1.0),
    (150, 0.10, 20, 1.0),
    (200, 0.05, 15, 2.0)
]

results = []

print("Training and evaluating six configurations...")

for max_iter, learning_rate, max_leaf_nodes, l2_regularization in configs:
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    results.append({
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
        "MAE": mae,
        "RMSE": rmse
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE").reset_index(drop=True)

print("\n========== HIST GRADIENT BOOSTING TUNING ==========")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nValidation period:")
print(df["date"].iloc[split_index])
print(f"{df['date'].iloc[-1]}")

best = results_df.iloc[0]

print("\nBest configuration:")
print(f"max_iter: {int(best['max_iter'])}")
print(f"learning_rate: {best['learning_rate']:.2f}")
print(f"max_leaf_nodes: {int(best['max_leaf_nodes'])}")
print(f"l2_regularization: {best['l2_regularization']:.1f}")
print(f"MAE: {best['MAE']:.6f}")
print(f"RMSE: {best['RMSE']:.6f}")