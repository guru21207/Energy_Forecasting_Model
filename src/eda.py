import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y %H:%M"
)

df = df.sort_values("date").reset_index(drop=True)

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget statistics:")
print(df["next_interval_Usage_kWh"].describe())

print("\nLoad type counts:")
print(df["Load_Type"].value_counts())

print("\nAverage usage by load type:")
print(df.groupby("Load_Type")["Usage_kWh"].mean())

print("\nNumeric correlations with target:")
print(
    df.corr(numeric_only=True)["next_interval_Usage_kWh"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 5))
plt.plot(
    df["date"],
    df["Usage_kWh"],
    linewidth=0.6
)
plt.title("Steel Plant Electricity Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Usage (kWh)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(
    column="Usage_kWh",
    by="Load_Type"
)
plt.title("Electricity Consumption by Load Type")
plt.suptitle("")
plt.xlabel("Load Type")
plt.ylabel("Usage (kWh)")
plt.tight_layout()
plt.show()

print("\nDuplicate rows:", df.duplicated().sum())

print("\nEarliest timestamp:", df["date"].min())
print("Latest timestamp:", df["date"].max())

print("\nTime interval differences:")
print(df["date"].diff().value_counts().head(10))

print("\nFirst 10 chronological records:")
print(
    df[
        ["date", "Usage_kWh", "Load_Type",
         "next_interval_Usage_kWh"]
    ].head(10).to_string(index=False)
)
df = df.sort_values("date").reset_index(drop=True)

df["next_usage_check"] = df["Usage_kWh"].shift(-1)

df["time_gap"] = df["date"].shift(-1) - df["date"]

valid_next = df["time_gap"] == pd.Timedelta(minutes=15)

print("Rows followed by a 15-minute interval:", valid_next.sum())

print(
    "Target matches next row's usage:",
    (
        df.loc[valid_next, "next_interval_Usage_kWh"].to_numpy()
        == df.loc[valid_next, "next_usage_check"].to_numpy()
    ).mean()
)

print("\nNon-15-minute gaps:")
print(df.loc[~valid_next, ["date", "time_gap"]].head(10))