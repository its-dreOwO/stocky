import pandas as pd
import numpy as np
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Load local data
data_path = os.path.join("data", "main_data", "tech_macro_aligned.csv")
print(f"Đang tải dữ liệu Apple (AAPL) từ {data_path}...")
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Filter for AAPL
aapl_df = df[df['Ticker'] == 'AAPL'].sort_index()
series = aapl_df['Close'].dropna()

log_returns = (np.log(series / series.shift(1)) * 100).dropna()

# Splitting data
train = log_returns.loc[:'2023-12-31']
val = log_returns.loc['2024-01-01':'2024-12-31']
# Test for 1 month starting from Feb 2025
test = log_returns.loc['2025-02-01':'2025-03-01']

test_prices = series.loc['2025-02-01':'2025-03-01']
# Use the price right before the test period as base
last_val_price = series.loc[:'2025-01-31'].iloc[-1]

history_returns = list(log_returns.loc[:'2025-01-31'].values)
actual_test_returns = list(test.values)

predicted_returns = []
predicted_volatility = []

print(f"--- Đang huấn luyện mô hình ARIMA-GARCH ({len(actual_test_returns)} ngày) ---")

for t in range(len(actual_test_returns)):
    # ARIMA(5,0,0) as in original script
    model_arima = ARIMA(history_returns, order=(5, 0, 0))
    model_arima_fit = model_arima.fit()
    
    arima_forecast = model_arima_fit.forecast()[0]
    predicted_returns.append(arima_forecast)
    
    residuals = model_arima_fit.resid
    model_garch = arch_model(residuals, vol='Garch', p=1, q=1, rescale=False)
    model_garch_fit = model_garch.fit(disp='off')
    
    garch_forecast = model_garch_fit.forecast(horizon=1)
    pred_variance = garch_forecast.variance.values[-1, 0]
    predicted_volatility.append(np.sqrt(pred_variance))
    
    history_returns.append(actual_test_returns[t])

predictions = []
actual_prices = test_prices.values

for t in range(len(predicted_returns)):
    if t == 0:
        base_price = last_val_price
    else:
        base_price = actual_prices[t-1] 
        
    pred_price = base_price * np.exp(predicted_returns[t] / 100)
    predictions.append(pred_price)

print("\n================================================================================")
print("BẢNG KIỂM NGHIỆM DỮ LIỆU THỰC TẾ (THÁNG 2/2025)")
print("================================================================================")
print(f"{'Date':<15} {'Thực tế':<15} {'Dự báo':<15} {'Sai lệch':<15}")

for i in range(len(test_prices)):
    date_str = test_prices.index[i].strftime('%Y-%m-%d')
    act = actual_prices[i]
    pred = predictions[i]
    err = pred - act
    print(f"{date_str:<15} {act:<15.2f} {pred:<15.2f} {err:<15.2f}")

print("\n--- ĐÁNH GIÁ HIỆU SUẤT ---")
avg_price = np.mean(actual_prices)
rmse = (np.sqrt(mean_squared_error(actual_prices, predictions)) / avg_price) * 100
print(f'1. RMSE: {rmse:.2f}%')

errors = np.abs(np.array(actual_prices) - np.array(predictions))
top_k_percent = 0.2
k = max(1, int(len(errors) * top_k_percent)) 
shock_mae = (np.mean(sorted(errors, reverse=True)[:k]) / avg_price) * 100
print(f'2. MAE (Top 20%): {shock_mae:.2f}%')

std_error = (np.std(errors) / avg_price) * 100
print(f'3. Standard Deviation (Errors): {std_error:.2f}%')

upper_bounds = []
lower_bounds = []
anomalies_x = []
anomalies_y = []

for t in range(len(predictions)):
    if t == 0:
        base_price = last_val_price
    else:
        base_price = actual_prices[t-1]
        
    upper_return = predicted_returns[t] + 1.96 * predicted_volatility[t]
    lower_return = predicted_returns[t] - 1.96 * predicted_volatility[t]
    
    upper_price = base_price * np.exp(upper_return / 100)
    lower_price = base_price * np.exp(lower_return / 100)
    
    upper_bounds.append(upper_price)
    lower_bounds.append(lower_price)
    
    if actual_prices[t] > upper_price or actual_prices[t] < lower_price:
        anomalies_x.append(test_prices.index[t])
        anomalies_y.append(actual_prices[t])

plt.figure(figsize=(12, 6))

# Updated colors to match eval_rolling_forcast.py
# Ground Truth: #060c8f, Forecast: #e74c3c
plt.plot(test_prices.index, actual_prices, color='#060c8f', marker='o', linewidth=2, label='Ground Truth')
plt.plot(test_prices.index, predictions, color='#e74c3c', linestyle='--', marker='s', label='Forecast (ARIMA/GARCH)')

plt.fill_between(test_prices.index, lower_bounds, upper_bounds, color='gray', alpha=0.1, label='Safe area GARCH (95%)')

if anomalies_x:
    plt.scatter(anomalies_x, anomalies_y, color='red', s=150, zorder=5, label='PHÁT HIỆN BẤT THƯỜNG (Shock)')

plt.title('Forecast Comparison: AAPL')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()