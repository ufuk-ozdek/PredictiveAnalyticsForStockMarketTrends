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


## **Requirements**
+ Python 3.x
+ pandas
+ numpy
+ matplotlib
+ scikit-learn
+ yfinance
+ shap

## Future Work
+ Integrate additional machine learning models like LSTM.
+ Extend the project to forecast multiple stocks simultaneously.
+ Deploy the model via a web application using Flask or Django.
