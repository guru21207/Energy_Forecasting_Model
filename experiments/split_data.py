import pandas as pd

df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y %H:%M"
)

df = df.sort_values("date").reset_index(drop=True)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
val_df = df.iloc[split_index:].copy()

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))

print("\nTraining period:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nValidation period:")
print(val_df["date"].min(), "to", val_df["date"].max())