# **Predictive Analytics For Stock Market Trends**

## **Project Description**

This project aims to build a predictive model for stock market trends using historical data. By leveraging Support Vector Regression (SVR) and advanced feature engineering, the project forecasts future stock prices for major companies like AAPL, MSFT, and GOOGL. The model is trained on data from 2020 to 2023 and incorporates features such as moving averages, momentum, and volatility.


## **Features**
+ **Data Collection:** Historical stock data fetched using the Yahoo Finance API.
+ **Preprocessing:** Handling missing data, outlier capping, and normalization.
+ **Feature Engineering:** Calculation of moving averages, momentum, volatility, and RSI.
+ **Modeling:** Tuning SVR with Grid Search for optimal performance.
+ **Evaluation:** Cross-validated RMSE, MSE, R-squared, and MAE metrics.
+ **Visualization:** Actual vs Predicted Prices, Residuals, and Feature Importance.

## Installation

To run this project locally, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ufuk-ozdek/PredictiveAnalyticsForStockMarketTrends.git
cd PredictiveAnalyticsForStockMarketTrends
pip install -r requirements.txt
```

## **Requirements**
+ Python 3.x
+ pandas
+ numpy
+ matplotlib
+ scikit-learn
+ yfinance
+ shap

## **Results**

- **AAPL**
  - Cross-Validated RMSE: 0.020787572189560993
  - MSE: 0.00047610138101505836
  - R-squared: 0.9410327633319528
  - MAE: 0.01619349333426411

![Actual vs Predicted Stock Prices - APPL](https://github.com/user-attachments/assets/189f921f-feb6-4968-8356-961b57635ab3)
![Residuals Over Time for APPL](https://github.com/user-attachments/assets/7f09141e-e7fa-4c7e-a954-6df5612906dd)
![SHAP Summary Plot for SVR Model Feature Importance](https://github.com/user-attachments/assets/3553b856-b387-4fc6-95ae-04920944e4cc)


- **MSFT**
  - Cross-Validated RMSE: 0.02198693771508212
  - MSE: 0.000721419433017819
  - R-squared: 0.9011299167486935
  - MAE: 0.020551217144603215
- **GOOGL**
  - Cross-Validated RMSE: 0.020347820042779595
  - MSE: 0.0005246558453682764
  - R-squared: 0.9503088334942307
  - MAE: 0.018224233848984688

  
## Future Work
+ Integrate additional machine learning models like LSTM.
+ Extend the project to forecast multiple stocks simultaneously.
+ Deploy the model via a web application using Flask or Django.
