import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
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

categorical_features = [
    "WeekStatus",
    "Day_of_week",
    "Load_Type"
]

numerical_features = [
    feature for feature in features
    if feature not in categorical_features
]

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
val_df = df.iloc[split_index:].copy()

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "HistGradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

    except Exception as error:
        print(f"\n{name} failed: {error}")

baseline_predictions = val_df["Usage_kWh"]

results.append({
    "Model": "Persistence Baseline",
    "MAE": mean_absolute_error(y_val, baseline_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_val, baseline_predictions))
})

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nTraining rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Validation starts:", val_df["date"].min())
print("Validation ends:", val_df["date"].max())