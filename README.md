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

### - **AAPL**
  - Cross-Validated RMSE: 0.020787572189560993
  - MSE: 0.00047610138101505836
  - R-squared: 0.9410327633319528
  - MAE: 0.01619349333426411


<img src="https://github.com/user-attachments/assets/1fa2c057-382f-47d7-9bf7-28c69e60e105" alt="Residuals Over Time - APPL" width="500"/>

<img src="https://github.com/user-attachments/assets/510af976-45b8-49ab-a8cc-8fde0ad0c97f" alt="Residuals Over Time - APPL" width="500"/>

<img src="https://github.com/user-attachments/assets/f8bb48df-9a49-4bab-a4bf-b311b254be61" alt="Residuals Over Time - APPL" width="500"/>




### - **MSFT**
  - Cross-Validated RMSE: 0.02198693771508212
  - MSE: 0.000721419433017819
  - R-squared: 0.9011299167486935
  - MAE: 0.020551217144603215



 
<img src="https://github.com/user-attachments/assets/a35f4c88-e81a-469d-8266-d351eec3b0ad" alt="Actual vs Predicted Stock Prices - MSFT" width="500"/>

<img src="https://github.com/user-attachments/assets/a92249a0-ce5d-4e58-a09a-3a709d46f424" alt="Residuals Over Time - MSFT" width="500"/>

<img src="https://github.com/user-attachments/assets/4f99e02d-48dc-470e-a5aa-8ae57e6fdbe5" alt="SHAP Summary Plot for SVR Model Feature Importance - MSFT" width="500"/>



### - **GOOGL**
  - Cross-Validated RMSE: 0.020347820042779595
  - MSE: 0.0005246558453682764
  - R-squared: 0.9503088334942307
  - MAE: 0.018224233848984688





 
<img src="https://github.com/user-attachments/assets/29f822a4-3af0-4d49-8ab7-bdb3eb9c2e9c" alt="Actual vs Predicted Stock Prices - GOOGL" width="500"/>

<img src="https://github.com/user-attachments/assets/bde67290-05e1-406a-9270-6bd70c387607" alt="Residuals Over Time - GOOGL" width="500"/>

<img src="https://github.com/user-attachments/assets/5213c9aa-f7c1-4e39-8a21-762bf9f0cdc3" alt="SHAP Summary Plot for SVR Model Feature Importance - GOOGL" width="500"/>



  
## Future Work
+ Integrate additional machine learning models like LSTM.
+ Extend the project to forecast multiple stocks simultaneously.
+ Deploy the model via a web application using Flask or Django.
