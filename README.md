# Steel Plant Energy Consumption Forecasting

## Project Overview

This project focuses on forecasting next-interval energy consumption in a steel plant using machine learning. It uses historical energy usage, electrical measurements, load types, and time-based features to predict future energy consumption.

The objective is to explore how machine learning-based forecasting can support industrial energy monitoring, consumption planning, and more informed energy-management decisions.

## Objectives

* Analyse historical industrial energy consumption data.
* Explore consumption patterns across different load types and time periods.
* Engineer time-based and historical energy usage features.
* Develop a machine learning regression model for next-interval energy forecasting.
* Evaluate forecasting performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* Analyse prediction errors across load types and time periods.
* Compare different modelling approaches and save the final trained model.

## Dataset

The project uses an industrial energy consumption dataset containing measurements recorded at regular time intervals.

The dataset includes the following information:

* **Date:** Timestamp associated with each observation.
* **Usage_kWh:** Recorded energy consumption.
* **Lagging_Current_Reactive.Power_kVarh:** Lagging reactive power measurement.
* **Leading_Current_Reactive_Power_kVarh:** Leading reactive power measurement.
* **CO2(tCO2):** Carbon dioxide emissions measurement.
* **Lagging_Current_Power_Factor:** Lagging power factor.
* **Leading_Current_Power_Factor:** Leading power factor.
* **NSM:** Number of seconds from midnight.
* **WeekStatus:** Indicates whether the observation belongs to a weekday or weekend.
* **Day_of_week:** Day of the week.
* **Load_Type:** Energy load category.
* **next_interval_Usage_kWh:** Target variable representing energy consumption for the next interval.

The dataset is stored in `data/train.csv`.

A cleaned base snapshot is also available at [`data/cleaned_train.csv`](data/cleaned_train.csv). It is generated in the notebook after parsing the timestamp and sorting the records chronologically. This file preserves the cleaned, timestamp-ordered base data; it does **not** contain the later engineered lag features used by the model. To regenerate it, run the corresponding save cell in `notebooks/energy_forecasting.ipynb`.

See the [Methodology Summary](docs/methodology_summary.md) for a concise overview of the EDA findings, feature engineering, chronological validation, business implications, interpretability, and limitations.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook

## Methodology

### 1. Data Inspection and Cleaning

The dataset is examined for missing values, duplicate records, and data type inconsistencies. The observations are sorted chronologically to maintain the correct time sequence.

### 2. Exploratory Data Analysis

Exploratory analysis is performed to understand the distribution of energy consumption and identify differences between load types. Daily average consumption is also visualised to explore changes over time.

### 3. Feature Engineering

Additional features are created to help the model identify temporal patterns and historical energy usage.

These include:

* Hour, minute, day, month, and day of the year.
* Day-of-week and weekend indicators.
* Cyclical time features using sine and cosine transformations.
* Interaction between hour and weekend status.
* Historical energy usage features representing previous 15-minute, 1-hour, and 24-hour intervals.

These features provide information about recent consumption and recurring temporal patterns.

### 4. Chronological Train–Validation Split

The dataset is divided into training and validation sets in chronological order.

Earlier observations are used for training, while later observations are reserved for validation. This approach better reflects forecasting on future observations than a randomly shuffled split.

### 5. Model Training

The primary forecasting model is a `HistGradientBoostingRegressor` from Scikit-learn.

A preprocessing pipeline handles numerical and categorical features:

* Missing numerical values are handled using median imputation.
* Missing categorical values are handled using the most frequent category.
* Categorical features are encoded using one-hot encoding.

The target variable is log-transformed during training to investigate whether reducing the influence of highly skewed target values improves prediction performance. Predictions are transformed back to the original energy consumption scale.

### 6. Model Evaluation

The model is evaluated using two regression metrics:

* **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted energy consumption.
* **Root Mean Squared Error (RMSE):** Measures prediction error while penalising larger errors more heavily.

Both metrics are calculated on the chronological validation set.

### 7. Error Analysis

Prediction errors are analysed across different load types and time periods to identify patterns in model performance.

This analysis helps highlight conditions under which forecasting errors may be larger and where further improvements could be explored.

### 8. Model Comparison and Experiments

The `experiments/` directory contains scripts used to investigate different modelling approaches and analyse forecasting performance.

These experiments include baseline evaluation, gradient boosting, log-transformed targets, quantile-based modelling, maximum-load analysis, and time-based error analysis.

### 9. Final Model Saving

After validation, the model is retrained using all available labelled observations and saved for future use.

The trained model and its feature list are stored in the `models/` directory.

## Project Structure

```text
Energy_Forecasting_Model/
├── data/
│   └── train.csv
├── experiments/
│   ├── base.py
│   ├── compare_models.py
│   ├── error_analysis.py
│   ├── histgradient_boost.py
│   ├── log_target_model.py
│   ├── maximum_load_model.py
│   ├── plot_model_comparison.py
│   ├── plot_results.py
│   ├── plot_time_errors.py
│   ├── quantile_model.py
│   ├── split_data.py
│   └── time_series_analysis.py
├── models/
│   ├── features.pkl
│   └── final_model.pkl
├── notebooks/
│   └── energy_forecasting.ipynb
├── plots/
│   ├── actual_vs_predicted_validation.png
│   ├── actual_vs_predicted.png
│   ├── error_by_load_type.png
│   ├── error_by_time.png
│   └── model_comparison.png
├── results/
│   ├── error_analysis.csv
│   └── time_error_analysis.csv
├── src/
│   ├── eda.py
│   └── train_final_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Directory Description

| Directory/File     | Description                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `data/`            | Contains the dataset used for training and analysis.                                                              |
| `experiments/`     | Contains scripts for baseline evaluation, alternative modelling approaches, model comparison, and error analysis. |
| `models/`          | Stores the final trained model and feature list.                                                                  |
| `notebooks/`       | Contains the main notebook documenting the forecasting workflow.                                                  |
| `plots/`           | Stores visualisations of predictions, model comparisons, and prediction errors.                                   |
| `results/`         | Contains CSV files with error-analysis results.                                                                   |
| `src/`             | Contains the main scripts for exploratory data analysis and final model training.                                 |
| `requirements.txt` | Lists the Python dependencies required by the project.                                                            |
| `.gitignore`       | Specifies files and directories that Git should ignore.                                                           |

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible version supported by the installed dependencies.
* Git.
* Jupyter Notebook or Visual Studio Code with Jupyter support.

### 1. Clone the Repository

Replace the example URL with the actual GitHub repository URL.

```bash
git clone https://github.com/guru21207/Energy_Forecasting_Model.git
cd Energy_Forecasting_Model
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Notebook

Open the project in Visual Studio Code or Jupyter Notebook.

Open:

```text
notebooks/energy_forecasting.ipynb
```

Select the Python environment where the dependencies are installed, then execute the notebook cells in order.

The notebook covers dataset inspection, exploratory analysis, feature engineering, chronological splitting, model training, evaluation, and error visualisation.

## Outputs

The project generates the following outputs:

* Actual-versus-predicted energy consumption graphs.
* Prediction error summaries by load type.
* Error analysis across time periods.
* Model comparison visualisations.
* CSV files containing error-analysis results.
* A saved final forecasting model.
* A saved list of model features.

## Limitations and Future Improvements

* Forecasting performance depends on the quality and coverage of historical data.
* Prediction errors may vary across load types and operating conditions.
* Additional features and alternative models could be investigated to improve forecasting accuracy.
* Evaluation across different time periods would help assess how consistently the model generalises.
* Further investigation could explore the relationship between energy consumption, operating conditions, and energy efficiency.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for industrial energy consumption forecasting, including data exploration, feature engineering, chronological validation, model training, performance evaluation, error analysis, and model saving.

The resulting workflow provides a foundation for further research into data-driven energy monitoring and forecasting in industrial environments.
