import pandas as pd
import numpy as np
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product

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


train_val = log_returns.loc[:'2024-12-31']
test = log_returns.loc['2025-02-01':'2025-03-01']
test_prices = series.loc['2025-02-01':'2025-03-01']




last_val_price = series.loc[:'2025-01-31'].iloc[-1]

def find_best_arima_order(data):
    p_range = range(0, 6)
    d_range = range(0, 2)
    q_range = range(0, 3)
    
    best_aic = float("inf")
    best_order = (5, 0, 0) # Default fallback
    
    print("--- Tìm tham số ARIMA tối ưu (Grid Search) ---")
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except:
            continue
    print(f"Tham số ARIMA tốt nhất: {best_order} (AIC: {best_aic:.2f})")
    return best_order

# Find best parameters once on training data
best_p, best_d, best_q = find_best_arima_order(train_val)

history_returns = list(log_returns.loc[:'2025-01-31'].values)
actual_test_returns = list(test.values)

predicted_returns = []
predicted_volatility = []

print(f"--- Đang huấn luyện mô hình ARIMA{best_p, best_d, best_q}-GARCH ({len(actual_test_returns)} ngày) ---")

for t in range(len(actual_test_returns)):
    model_arima = ARIMA(history_returns, order=(best_p, best_d, best_q))
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
print(f"BẢNG KIỂM NGHIỆM DỮ LIỆU THỰC TẾ (THÁNG 2/2025) - ARIMA{best_p, best_d, best_q}")
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
print(f'> RMSE: {rmse:.2f}%')

errors = np.abs(np.array(actual_prices) - np.array(predictions))
top_k_percent = 0.2
k = max(1, int(len(errors) * top_k_percent)) 
shock_mae = (np.mean(sorted(errors, reverse=True)[:k]) / avg_price) * 100
print(f'> MAE: {shock_mae:.2f}%')

std_error = (np.std(errors) / avg_price) * 100
print(f'> STD: {std_error:.2f}%')

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

plt.plot(test_prices.index, actual_prices, color='#060c8f', marker='o', linewidth=2, label='Ground Truth')
plt.plot(test_prices.index, predictions, color='#e74c3c', linestyle='--', marker='s', label=f'Forecast (ARIMA{best_p, best_d, best_q}/GARCH)')

plt.fill_between(test_prices.index, lower_bounds, upper_bounds, color='gray', alpha=0.1, label='Safe area GARCH (95%)')

if anomalies_x:
    plt.scatter(anomalies_x, anomalies_y, color='red', s=150, zorder=5, label='Shock detected')

plt.title(f'Forecast Comparison: AAPL (ARIMA/GARCH)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()