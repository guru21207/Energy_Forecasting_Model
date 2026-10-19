# Methodology Summary: Steel Plant Energy Consumption Forecasting

## 1. Objective and dataset
The objective is to estimate energy consumption for the next interval using historical energy readings and contextual information. The supplied steel-plant dataset contains **28,031 rows and 12 columns**. The timestamp was parsed using the dataset's day/month/year format, and records were sorted chronologically. Initial checks found **no missing values and no duplicate rows** in the original dataset.

The prediction target is `next_interval_Usage_kWh`. Exploratory analysis examined the target distribution, usage across `Load_Type`, and daily average usage over time.

## 2. Key EDA findings
The notebook visualizes the distribution of the prediction target, compares usage across load types, and plots daily average usage to inspect temporal patterns. Validation errors differed by load type: **Light_Load: 1.1689 kWh MAE; Medium_Load: 4.8455 kWh MAE; Maximum_Load: 8.9442 kWh MAE**. This indicates that predictions were less accurate for maximum-load periods in the observed validation set. These results do not establish the cause of the differences.

## 3. Feature engineering and model choices
Timestamp-derived features include hour, minute, day, month, day of year, day of week, and a weekend indicator. Sine/cosine encodings represent cyclical hour and day patterns, and an hour–weekend interaction was added.

Three lag features were created from historical `Usage_kWh`: approximately **15 minutes, 60 minutes, and 24 hours** before the target timestamp. Rows without the required lag history were removed from the modelling dataset. The raw timestamp and target were excluded from the feature matrix.

A scikit-learn pipeline handles numeric missing values with median imputation and categorical missing values with most-frequent imputation, followed by one-hot encoding. The regressor is `HistGradientBoostingRegressor` with `max_iter=250`, `learning_rate=0.05`, `max_leaf_nodes=31`, and `random_state=42`. The target was transformed using `log1p` during training and predictions were converted back using `expm1`; negative predictions were clipped to zero.

## 4. Validation approach and results
The data was split **chronologically**, using the earlier 80% for training and the later 20% for validation. This reflects forecasting future observations better than a random split that mixes earlier and later timestamps. Performance was evaluated using mean absolute error (MAE) and root mean squared error (RMSE).

- **Validation MAE:** 3.7954 kWh
- **Validation RMSE:** 8.5569 kWh

The notebook plots actual versus predicted values and compares errors by load type. After validation, the pipeline is retrained on all available labelled data and saved as `models/final_model.pkl`; feature names are saved as `models/features.pkl`.

## 5. Practical business takeaways
The forecast can support short-term energy monitoring, planning, and identifying intervals that may need closer operational review. The higher validation error for `Maximum_Load` suggests that high-demand periods deserve particular attention before operational decisions rely on the prediction. The model should support, not replace, plant monitoring and operator judgement.

## 6. Interpretability and limitations
Interpretability in the current notebook is based mainly on actual-versus-predicted visualisation and error breakdown by load type. These show where predictions are inaccurate, but do **not** identify which individual features drive predictions. Formal feature-importance analysis or SHAP explanations have not been included.

The reported metrics come from one chronological holdout, so performance may vary across other periods or operating conditions. Further evaluation could include rolling time-based validation, comparison with a simple baseline, and feature-importance analysis. These results show predictive performance on the available data; they do not demonstrate causal relationships or guaranteed energy savings.

## Reproducibility
The workflow is documented in [`notebooks/energy_forecasting.ipynb`](../notebooks/energy_forecasting.ipynb). Required Python packages are listed in [`requirements.txt`](../requirements.txt). The cleaned base dataset is available at [`data/cleaned_train.csv`](../data/cleaned_train.csv).
