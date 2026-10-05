# Delhi-Daily-Temperature-Forecasting-with-a-Stacked-LSTM
A deep learning time-series project that forecasts the next-day mean temperature in Delhi using a two-layer LSTM recurrent neural network.

## 📌 Overview

The project explores weather trends and seasonality before building a recurrent neural network for temperature forecasting.

The workflow includes:

* Exploratory data analysis
* Chronological train/validation/test splitting
* Feature engineering
* Data scaling
* 30-day sliding-window sequences
* Stacked LSTM training
* Model evaluation and analysis

## 🧹 Data Preparation

* Converted dates to `datetime`
* Split the training data chronologically into 80% training and 20% validation
* Kept the provided test set separate
* Added `month` and `day` features
* Fitted `MinMaxScaler` only on the training data to prevent data leakage
* Created 30-day sliding windows to predict the next day's temperature
* Kept wind-speed outliers because they may represent genuine weather events

The model uses four weather features:

* `meantemp`
* `humidity`
* `wind_speed`
* `meanpressure`

### Dataset

| Dataset                      |  Rows | Description                            |
| ---------------------------- | ----: | -------------------------------------- |
| `DailyDelhiClimateTrain.csv` | 1,462 | Daily records starting from 2013-01-01 |
| `DailyDelhiClimateTest.csv`  |   114 | Daily records starting from 2017-01-01 |

Sequence sizes:

| Dataset    | Sequences |
| ---------- | --------: |
| Training   |     1,139 |
| Validation |       263 |
| Test       |        84 |


## 📊 Exploratory Data Analysis

The analysis includes:

* Temperature seasonality and trends
* 30-day moving averages
* Numerical feature distributions
* Outlier detection
* Feature correlation analysis

The strongest correlations in the training data were:

| Feature Pair             | Correlation |
| ------------------------ | ----------: |
| Temperature & Pressure   |       -0.89 |
| Temperature & Humidity   |       -0.59 |
| Temperature & Wind Speed |       +0.30 |

The dataset also contains several anomalous pressure values outside the normal range, particularly in the validation and test data.

## 🤖 Model

The model uses two stacked LSTM layers:

```text
Input (30 days × 4 features)
        ↓
LSTM(32, return_sequences=True)
        ↓
Dropout(0.2)
        ↓
LSTM(16)
        ↓
Dropout(0.2)
        ↓
Dense(1)
```

**Training configuration:**

* Adam optimizer
* Mean Squared Error (MSE) loss
* Batch size: 32
* Maximum epochs: 30
* EarlyStopping with patience of 5
* 7,889 trainable parameters

## 📈 Results

Evaluated on **84 test predictions**:

| Metric |      Result |
| ------ | ----------: |
| MAE    | **2.18 °C** |
| MSE    |    **6.83** |
| RMSE   | **2.61 °C** |
| R²     |   **0.803** |

### Key Findings

* The model explains approximately **80% of the variance** in the test temperatures.
* Predictions are typically around **2.2 °C** away from the actual temperature.
* The model successfully follows the overall seasonal trend.
* Predictions are smoother than the actual data and struggle with sudden daily temperature changes.
* Training and validation loss remained close, showing no clear signs of overfitting.
* The model's best validation loss was **0.0063** around epoch 27.

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras** — stacked LSTM and EarlyStopping
* **scikit-learn** — scaling and evaluation metrics
* **pandas & NumPy** — data processing
* **Matplotlib & Seaborn** — visualization

## 📁 Project Structure

```text
Delhi-Temperature-Forecasting/
├── StackedLSTMClimateChangeDELHI.ipynb
├── DailyDelhiClimateTrain.csv
├── DailyDelhiClimateTest.csv
└── README.md
```

## 🔮 Possible Improvements

* Handle anomalous pressure values before scaling
* Incorporate the `month` and `day` features into the model
* Compare against a simple baseline such as predicting tomorrow's temperature as today's
* Carry previous split data into validation and test windows
* Experiment with GRU, deeper LSTM, or longer training

*Built as a deep learning project exploring weather time-series forecasting with stacked LSTM networks.*
