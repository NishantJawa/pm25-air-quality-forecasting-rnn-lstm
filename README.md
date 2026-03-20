# PM2.5 Air Quality Forecasting using Deep Learning (RNN, LSTM, GRU)

## Project Overview

This project focuses on forecasting hourly PM2.5 air pollution levels in Beijing using deep learning-based time series models. The objective is to develop a predictive system that can provide accurate short-term air quality forecasts, supporting early-warning systems and data-driven environmental decision-making.

The dataset includes hourly pollution and meteorological data from 2013 to 2016. A structured modelling approach was followed, starting from baseline recurrent models and progressively moving toward more advanced architectures. Multiple models were developed and compared, including SimpleRNN, LSTM, and GRU variants.

The final selected model was a **2-layer LSTM (128 → 64 units with dropout)**, which demonstrated the best performance across all evaluation metrics.


## Business Problem

Air pollution, particularly PM2.5 concentration, has a direct impact on public health and urban living conditions. Governments and environmental agencies require reliable forecasting systems to anticipate pollution spikes and take proactive measures such as issuing alerts, regulating traffic, or controlling industrial emissions.

Traditional statistical models often struggle to capture the complex temporal dependencies present in air quality data. This project addresses that limitation by using deep learning models capable of learning both short-term fluctuations and long-term seasonal patterns.

The goal is to build a model that can forecast PM2.5 levels for the next 24 hours, enabling timely and informed decision-making.


## Dataset and Preprocessing

The dataset consists of hourly observations of air pollutants and weather conditions, including:

- PM2.5 (target variable)
- SO₂, NO₂, CO, O₃
- Temperature, Pressure, Dew Point, Wind Speed

The data spans from 2013 to 2016 and contains over **35,000 observations**.

Several preprocessing steps were applied to ensure data quality and model readiness:

- Converted timestamps into a proper datetime index  
- Handled missing values using time-based interpolation (NO₂ and O₃)  
- Applied feature scaling using MinMaxScaler  
- Split data into:
  - Training set: 2013–2015  
  - Testing set: January 2016  

To frame the problem as a supervised learning task, a sliding window approach was used:
- **Input (lookback):** 72 hours  
- **Output (forecast horizon):** next 24 hours  

This allows the model to learn temporal relationships and predict future pollution levels based on recent patterns.


## Modelling Approach

A structured and progressive experimentation strategy was followed instead of testing random models.

### Baseline Models
Initial models included:
- SimpleRNN (100 units)
- LSTM (100 units)
- GRU (100 units)

These models established a performance benchmark and helped understand how different architectures handle sequential data.

### Reduced and Lightweight Models
Simpler models with fewer parameters were tested to evaluate whether reducing complexity improves generalisation:
- SimpleRNN (50 units)
- LSTM with fewer training epochs

### Stacked Architectures
Deeper models were introduced to capture more complex temporal patterns:
- Stacked SimpleRNN (128 → 64)
- Alternative optimisers (Adam vs RMSprop)

### Advanced Models with Regularisation
Finally, more robust architectures were developed using:
- Dropout and recurrent dropout  
- Multi-layer structures  
- Dense layers for nonlinear mapping  

This stage led to the best-performing models, particularly LSTM and GRU variants.

---

## Final Model

The best model was a **2-layer LSTM architecture (128 → 64 units)** with dropout and a dense output layer.

### Architecture Summary
- LSTM (128 units, return sequences)
- LSTM (64 units)
- Dropout + recurrent dropout
- Dense (64, ReLU)
- Output layer (24-step forecast)

This model was designed to:
- capture long-term dependencies (via LSTM memory)
- reduce overfitting (via dropout)
- map learned features to multi-step predictions

---

## Model Performance

The final model achieved:

- **RMSE:** ~44.3  
- **MAE:** ~34.2  
- **R²:** ~0.32  

Compared to baseline models, the 2-layer LSTM significantly improved performance by capturing both short-term fluctuations and longer-term trends in pollution levels.

An important observation is that prediction accuracy decreases as the forecasting horizon increases:
- very accurate for **1–6 hours ahead**
- gradually less reliable beyond **12–24 hours**

This behaviour is expected in time series forecasting and reflects increasing uncertainty over time.


## Key Visualisations

### Time Series Trends
These plots show the behaviour of pollutants and meteorological variables over time, highlighting seasonality, volatility, and trends.

![Time Series](images/time_series_all_features.png)


### Training Sample (72h → 24h Forecast)
This visual explains how the model learns from past data to predict future PM2.5 values.

<<img width="1089" height="590" alt="image" src="https://github.com/user-attachments/assets/b363925a-a54e-4567-b2fb-a2f88db130dd" />
<<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/cc58be16-0f3c-452a-9f93-7815bbb0180a" />


### Model Training Loss
This plot shows convergence of the LSTM model and helps assess training stability.

<img width="495" height="182" alt="Screenshot 2026-03-20 at 12 56 52 pm" src="https://github.com/user-attachments/assets/cb9c0414-cb92-493d-8d27-ade5da2fc7d3" />


### Prediction vs Actual (Single Sample)
This visual compares predicted vs actual PM2.5 values for a test sample, showing how well the model tracks real-world patterns.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/ee41382b-b853-4fa3-b935-3f5703ddf805" />


### Multi-Horizon Forecast Performance
This grid shows predictions across all 24 forecast horizons, highlighting how accuracy decreases over time.

<img width="1348" height="869" alt="image" src="https://github.com/user-attachments/assets/bb209250-a692-4c53-b18d-dc5c53e3a7a5" />


## Key Findings and Insights

The experimentation revealed several important insights:

The **SimpleRNN baseline surprisingly outperformed LSTM and GRU baselines**, indicating that simpler models can sometimes capture short-term dependencies effectively.

Reducing model complexity improved performance in some cases, suggesting that overfitting was a key challenge with larger models.

Stacked architectures without proper regularisation performed poorly, reinforcing the importance of dropout in deep learning models.

The **2-layer LSTM with dropout delivered the best results**, as it successfully balanced model complexity and generalisation.

From a forecasting perspective, the model is highly reliable for short-term predictions (1–6 hours ahead), making it suitable for real-world alert systems.


## Business Value and Applications

This model can be applied in real-world air quality monitoring systems to:

- provide early warnings for pollution spikes  
- support government policy decisions  
- assist healthcare advisories for vulnerable populations  
- optimise industrial and traffic control measures  

By enabling short-term forecasting, the system helps organisations move from reactive to proactive decision-making.


## Limitations and Future Improvements

Despite strong performance, there are several limitations:

Forecast accuracy decreases for longer horizons (beyond 12–24 hours), which limits long-term planning.

The model depends heavily on clean and continuous data, making data quality critical for deployment.

Future improvements could include:
- incorporating external factors (e.g., traffic, industrial activity)
- using advanced architectures (e.g., attention models, transformers)
- applying ensemble techniques
- increasing dataset size for better generalisation


## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Google Colab  
