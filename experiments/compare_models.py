import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)

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
    "is_weekend"
]

X = df[features]
y = df[target]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
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

baseline_predictions = X_val["Usage_kWh"]

results.append({
    "Model": "Persistence Baseline",
    "MAE": mean_absolute_error(y_val, baseline_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_val, baseline_predictions))
})

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))