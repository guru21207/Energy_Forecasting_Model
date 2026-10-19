import matplotlib.pyplot as plt
import numpy as np

models = ["Baseline", "Existing Model", "Log-Target Model"]
mae = [None, 4.0486, 3.8159]
rmse = [None, 8.3787, 8.4829]

mae[0] = 0
rmse[0] = 0

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - width / 2, mae, width, label="MAE")
ax.bar(x + width / 2, rmse, width, label="RMSE")

ax.set_ylabel("Error (kWh)")
ax.set_title("Energy Forecasting Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=300)
plt.show()
