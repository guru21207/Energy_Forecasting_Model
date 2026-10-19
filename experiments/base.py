import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y %H:%M"
)

df = df.sort_values("date").reset_index(drop=True)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
val_df = df.iloc[split_index:].copy()

y_true = val_df["next_interval_Usage_kWh"]
y_pred = val_df["Usage_kWh"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("Baseline Model Results")
print("----------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")