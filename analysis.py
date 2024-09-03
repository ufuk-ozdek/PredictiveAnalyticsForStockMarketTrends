import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import shap

def download_data(companies, start_date='2020-01-01', end_date='2023-01-01'):
    data = {}
    for company in companies:
        data[company] = yf.download(company, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler()
    for company in data:
        data[company].ffill(inplace=True)
        cap_outliers(data[company])
        data[company]['Close'] = scaler.fit_transform(data[company]['Close'].values.reshape(-1, 1))
    return data

def cap_outliers(df):
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Close'] = np.where(df['Close'] > upper_bound, upper_bound, np.where(df['Close'] < lower_bound, lower_bound, df['Close']))

def feature_engineering(data):
    for company in data:
        data[company]['10_MA'] = data[company]['Close'].rolling(window=10).mean()
        data[company]['100_MA'] = data[company]['Close'].rolling(window=100).mean()
        data[company]['Momentum'] = data[company]['Close'] - data[company]['Close'].shift(10)
        data[company]['Volatility'] = data[company]['Close'].rolling(window=10).std()
        data[company]['RSI'] = calculate_rsi(data[company]['Close'])
    return data

def calculate_rsi(close_prices, window_length=14):
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def parameter_tuning(data, companies):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1e-3, 1e-4, 'scale', 'auto'],
        'epsilon': [0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svr = SVR()

    for company in companies:
        X_company = data[company][['10_MA', '100_MA', 'Momentum', 'Volatility', 'RSI']]
        y_company = data[company]['Close']
        X_company = X_company.dropna()
        y_company = y_company.loc[X_company.index]

        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_search.fit(X_company, y_company)

        best_model = grid_search.best_estimator_
        evaluate_model(best_model, X_company, y_company, company)

def evaluate_model(model, X, y, company):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'{company} - SVR:')
    print(f'    Cross-Validated RMSE: {cv_rmse}')
    print(f'    MSE: {mse}')
    print(f'    R-squared: {r2}')
    print(f'    MAE: {mae}\n')

    plot_results(y_test, y_pred, company)
    residual_analysis(y_test, y_pred, company)
    plot_feature_importance(model, X_test)

def plot_results(y_test, y_pred, company):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Prices')
    plt.plot(y_test.index, y_pred, label='Predicted Prices (SVR)', linestyle='--')
    plt.title(f'{company} Actual vs Predicted Stock Prices (SVR)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()

def residual_analysis(y_test, y_pred, company):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, residuals, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals Over Time for {company}')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

def plot_feature_importance(model, X_test):
    explainer = shap.KernelExplainer(model.predict, X_test)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

def main():
    companies = ['AAPL', 'MSFT', 'GOOGL']
    data = download_data(companies)
    data = preprocess_data(data)
    data = feature_engineering(data)
    parameter_tuning(data, companies)

if __name__ == "__main__":
    main()
