import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from pathlib import Path

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

model = HistGradientBoostingRegressor(
    max_iter=250,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("Training the model...")

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

results = pd.DataFrame({
    "Date": df["date"].iloc[split_index:].values,
    "Load_Type": df["Load_Type"].iloc[split_index:].values,
    "Actual": y_val.values,
    "Predicted": predictions
})

results["Error"] = results["Predicted"] - results["Actual"]
results["Absolute_Error"] = results["Error"].abs()

print("\n========== ERROR ANALYSIS ==========")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

print("\nValidation period:")
print(results["Date"].min())
print(results["Date"].max())

print("\n========== 20 LARGEST ERRORS ==========")

largest_errors = results.nlargest(20, "Absolute_Error")

print(
    largest_errors[
        ["Date", "Load_Type", "Actual", "Predicted", "Error", "Absolute_Error"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\n========== ERROR BY LOAD TYPE ==========")

load_errors = results.groupby("Load_Type").agg(
    Rows=("Absolute_Error", "count"),
    MAE=("Absolute_Error", "mean"),
    Average_Error=("Error", "mean"),
    Maximum_Error=("Absolute_Error", "max")
).sort_values("MAE", ascending=False)

print(load_errors.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n========== ERROR SUMMARY ==========")
print(f"Mean error: {results['Error'].mean():.4f}")
print(f"Median absolute error: {results['Absolute_Error'].median():.4f}")
print(f"Maximum absolute error: {results['Absolute_Error'].max():.4f}")
print(f"Errors above 10: {(results['Absolute_Error'] > 10).sum()}")
print(f"Errors above 20: {(results['Absolute_Error'] > 20).sum()}")

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "error_analysis.csv"
results.to_csv(output_path, index=False)

print(f"\nError report saved to: {output_path}")