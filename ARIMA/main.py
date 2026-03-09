import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# 1. Data Fetching
tickers = ['^DJI', 'FEZ', 'VNM']
print("Đang tải dữ liệu Dow Jones (^DJI)...")
data = yf.download(tickers, start='2022-01-01', end='2025-12-31')['Close']
series = data['^DJI'].dropna()

# 2. Preprocessing: Convert to Log Returns (* 100)
# This solves the GARCH DataScaleWarning and prevents ARIMA straight-line drift
log_returns = (np.log(series / series.shift(1)) * 100).dropna()

# Split Data (using returns)
train = log_returns.loc['2022-01-01':'2023-12-31']
val = log_returns.loc['2024-01-01':'2024-12-31']
test = log_returns.loc['2025-01-01':'2025-12-31']

# Keep actual prices for reconstruction and evaluation later
test_prices = series.loc['2025-01-01':'2025-12-31']
last_val_price = series.loc[:'2024-12-31'].iloc[-1]

history_returns = list(train.values) + list(val.values)
actual_test_returns = list(test.values)

predicted_returns = []
predicted_volatility = []

print(f"--- Đang huấn luyện mô hình ARIMA-GARCH ({len(actual_test_returns)} ngày) ---")

# 3. Rolling Forecast Loop
for t in range(len(actual_test_returns)):
    # --- ARIMA (Mean Model) ---
    # Order is (5,0,0) because log returns are already stationary (d=0)
    model_arima = ARIMA(history_returns, order=(5, 0, 0))
    model_arima_fit = model_arima.fit()
    
    # Forecast 1-step ahead return
    arima_forecast = model_arima_fit.forecast()[0]
    predicted_returns.append(arima_forecast)
    
    # --- GARCH (Volatility Model) ---
    # Fit GARCH on the residuals (errors) of the ARIMA model
    residuals = model_arima_fit.resid
    model_garch = arch_model(residuals, vol='Garch', p=1, q=1, rescale=False)
    model_garch_fit = model_garch.fit(disp='off')
    
    # Forecast 1-step ahead variance
    garch_forecast = model_garch_fit.forecast(horizon=1)
    pred_variance = garch_forecast.variance.values[-1, 0]
    predicted_volatility.append(np.sqrt(pred_variance))
    
    # Update history with the ACTUAL observed return to step forward
    history_returns.append(actual_test_returns[t])

# 4. Reconstruct Prices from Predicted Returns
# Formula: P_t = P_{t-1} * exp(R_t / 100)
predictions = []
actual_prices = test_prices.values

for t in range(len(predicted_returns)):
    # To prevent compounding errors, our baseline is always yesterday's TRUE price
    if t == 0:
        base_price = last_val_price
    else:
        base_price = actual_prices[t-1] 
        
    pred_price = base_price * np.exp(predicted_returns[t] / 100)
    predictions.append(pred_price)

# 5. Evaluation
print("\n================================================================================")
print("BẢNG KIỂM NGHIỆM DỮ LIỆU THỰC TẾ (THÁNG 1/2025)")
print("================================================================================")
print(f"{'Date':<15} {'Thực tế':<15} {'Dự báo':<15} {'Sai lệch':<15}")

for i in range(min(10, len(test_prices))):
    date_str = test_prices.index[i].strftime('%Y-%m-%d')
    act = actual_prices[i]
    pred = predictions[i]
    err = pred - act
    print(f"{date_str:<15} {act:<15.2f} {pred:<15.2f} {err:<15.2f}")

print("\n--- ĐÁNH GIÁ HIỆU SUẤT (PERFORMANCE METRICS) ---")
# Metric phụ: RMSE Toàn cục 
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
print(f'1. RMSE (Sai số toàn cục): {rmse:.2f}')

# Metric chính: MAE Shock-focused 
errors = np.abs(np.array(actual_prices) - np.array(predictions))
top_k_percent = 0.2
k = max(1, int(len(errors) * top_k_percent)) # Ensure at least 1 item is selected
shock_mae = np.mean(sorted(errors, reverse=True)[:k])
print(f'2. MAE (Shock-focused Top {top_k_percent*100}%): {shock_mae:.2f}')
import matplotlib.pyplot as plt

# 6. Tính toán dải băng GARCH (GARCH Bounds) và Phát hiện Anomaly
upper_bounds = []
lower_bounds = []
anomalies_x = []
anomalies_y = []

for t in range(len(predictions)):
    # Lấy giá trị cơ sở của ngày hôm trước
    if t == 0:
        base_price = last_val_price
    else:
        base_price = actual_prices[t-1]
        
    # Tính khoảng tin cậy 95% cho lợi nhuận (Z-score = 1.96)
    upper_return = predicted_returns[t] + 1.96 * predicted_volatility[t]
    lower_return = predicted_returns[t] - 1.96 * predicted_volatility[t]
    
    # Chuyển đổi lợi nhuận trở lại thành giá trị Index (Price)
    upper_price = base_price * np.exp(upper_return / 100)
    lower_price = base_price * np.exp(lower_return / 100)
    
    upper_bounds.append(upper_price)
    lower_bounds.append(lower_price)
    
    # Phát hiện Anomaly: Nếu giá thực tế nằm ngoài dải băng GARCH
    if actual_prices[t] > upper_price or actual_prices[t] < lower_price:
        anomalies_x.append(test_prices.index[t])
        anomalies_y.append(actual_prices[t])

# 7. Vẽ biểu đồ (Matplotlib)
plt.figure(figsize=(12, 6))

# Đường giá thực tế (Màu xanh lá)
plt.plot(test_prices.index, actual_prices, color='#2ca02c', marker='o', linewidth=2, label='Thực tế (Actual)')

# Đường dự báo ARIMA (Màu xanh dương nét đứt)
plt.plot(test_prices.index, predictions, color='blue', linestyle='--', label='Dự báo ARIMA')

# Vùng an toàn GARCH (Màu đỏ nhạt)
plt.fill_between(test_prices.index, lower_bounds, upper_bounds, color='red', alpha=0.08, label='Vùng an toàn GARCH (95%)')

# Điểm bất thường / Shock (Chấm đỏ lớn)
if anomalies_x:
    plt.scatter(anomalies_x, anomalies_y, color='red', s=150, zorder=5, label='PHÁT HIỆN BẤT THƯỜNG (Shock)')

# Định dạng biểu đồ
plt.title('Kiểm nghiệm ARIMA-GARCH trên dữ liệu thực tế (Năm 2025)')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.2, linestyle='--')
plt.tight_layout()

# Hiển thị
plt.show()