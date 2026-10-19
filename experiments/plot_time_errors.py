import pandas as pd
import matplotlib.pyplot as plt

data = {
"Time Period": ["Night", "Morning", "Afternoon", "Evening"],
"MAE": [0.3751, 5.5686, 6.8110, 3.0948]
}

df = pd.DataFrame(data)

plt.figure(figsize=(9, 5))
bars = plt.bar(df["Time Period"], df["MAE"])

plt.title("Forecasting Error by Time of Day")
plt.xlabel("Time Period")
plt.ylabel("Mean Absolute Error (kWh)")

for bar, value in zip(bars, df["MAE"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("plots/error_by_time.png", dpi=300)
plt.show()
