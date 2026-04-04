import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import os

# Cài đặt giao diện đồ thị
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("--- BẮT ĐẦU QUY TRÌNH ---")

# =====================================================================
# BƯỚC 1: PANDAS & NUMPY - Khởi tạo và Khám phá dữ liệu thực tế
# =====================================================================
print("1. Đang tải dữ liệu thực tế (tech_macro_aligned.csv)...")

# Xác định đường dẫn file linh hoạt
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "..", "data", "main_data", "tech_macro_aligned.csv")

if not os.path.exists(data_path):
    print(f"LỖI: Không tìm thấy file tại {data_path}")
    # Fallback to current working directory if run from root
    data_path = os.path.join("data", "main_data", "tech_macro_aligned.csv")

full_df = pd.read_csv(data_path)

# Lọc dữ liệu cho ticker AAPL và chuyển Date sang datetime
df = full_df[full_df['Ticker'] == 'AAPL'].copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Tính Moving Average để làm mượt dữ liệu (Step 1 analysis)
df['MA20'] = df['Close'].rolling(window=20).mean()

# =====================================================================
# BƯỚC 2: SCIKIT-LEARN - Tiền xử lý (Chuẩn hóa dữ liệu)
# =====================================================================
print("2. Đang chuẩn hóa dữ liệu (Scikit-learn)...")
scaler = MinMaxScaler(feature_range=(0, 1))
df['Scaled_Close'] = scaler.fit_transform(df[['Close']])

# =====================================================================
# TRỰC QUAN HÓA BƯỚC 1 & 2 (1+2 STEP MODIFIED)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle('QUY TRÌNH BIẾN ĐỔI DỮ LIỆU (STEP 1 + 2)', fontsize=18, fontweight='bold', y=0.98)

# STEP 1: Dữ liệu thô và Xu hướng
ax1.plot(df.index, df['Close'], color='#1f77b4', label='Giá Close thực tế', alpha=0.6)
ax1.plot(df.index, df['MA20'], color='#d62728', label='Moving Average (20 ngày)', linewidth=2)
ax1.set_title('BƯỚC 1: Dữ liệu thô ban đầu (Raw Data)', fontsize=14, loc='left', fontweight='bold')
ax1.set_ylabel('Giá (USD)', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# STEP 2: Dữ liệu đã chuẩn hóa
ax2.plot(df.index, df['Scaled_Close'], color='#2ca02c', label='Dữ liệu MinMaxScaler (0-1)', linewidth=1.5)
ax2.set_title('BƯỚC 2: Dữ liệu sau khi Chuẩn hóa (Normalized Data)', fontsize=14, loc='left', fontweight='bold')
ax2.set_ylabel('Giá trị Scale', fontsize=12)
ax2.set_xlabel('Thời gian', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Thêm Histogram nhỏ ở bên cạnh (In-set plots)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Inset cho Step 1
ax_ins1 = inset_axes(ax1, width="20%", height="30%", loc='lower right', borderpad=2)
sns.histplot(df['Close'], kde=True, color='#1f77b4', ax=ax_ins1)
ax_ins1.set_title('Phân phối Raw', fontsize=10)
ax_ins1.set_xlabel('')
ax_ins1.set_ylabel('')

# Inset cho Step 2
ax_ins2 = inset_axes(ax2, width="20%", height="30%", loc='lower right', borderpad=2)
sns.histplot(df['Scaled_Close'], kde=True, color='#2ca02c', ax=ax_ins2)
ax_ins2.set_title('Phân phối Scaled', fontsize=10)
ax_ins2.set_xlabel('')
ax_ins2.set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =====================================================================
# BƯỚC 3: PYTORCH - Chuẩn bị Dataset & Xây dựng Mô hình
# =====================================================================
print("3. Đang chuẩn bị Tensor và Huấn luyện Mô hình (PyTorch)...")
# Tạo chuỗi dữ liệu (Sliding Window) cho PyTorch
SEQ_LEN = 20 # Tăng sequence length cho dữ liệu thực tế
X, y = [], []
scaled_data = df['Scaled_Close'].values

for i in range(len(scaled_data) - SEQ_LEN):
    X.append(scaled_data[i:i+SEQ_LEN])
    y.append(scaled_data[i+SEQ_LEN])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

# Chia tập Train/Test (80/20)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Định nghĩa mạng Neural Network đơn giản
class SimpleForecaster(nn.Module):
    def __init__(self):
        super(SimpleForecaster, self).__init__()
        self.linear1 = nn.Linear(SEQ_LEN, 32) # Tăng số neuron cho dữ liệu thực
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

model = SimpleForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Huấn luyện mô hình
epochs = 150 # Tăng số epoch cho dữ liệu thực
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Dự đoán trên tập Test
model.eval()
with torch.no_grad():
    test_preds = model(X_test)

# Đảo ngược Scale
actual_prices = scaler.inverse_transform(y_test.numpy())
predicted_prices = scaler.inverse_transform(test_preds.numpy())
mse_score = mean_squared_error(actual_prices, predicted_prices)

# =====================================================================
# BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ HUẤN LUYỆN (MATPLOTLIB)
# =====================================================================
print("4. Đang vẽ biểu đồ kết quả mô hình...")
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('BƯỚC 3 & 4: KẾT QUẢ HUÂN LUYỆN VÀ DỰ BÁO AAPL', fontsize=16, fontweight='bold')

# 3.1 Biểu đồ Loss Curve
axs[0].plot(range(epochs), losses, color='red', linewidth=2)
axs[0].set_title('Quá trình hội tụ của Loss (MSE)', fontsize=13)
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_yscale('log') # Dùng log scale để thấy rõ sự hội tụ

# 4.1 Biểu đồ Dự đoán vs Thực tế
test_dates = df.index[train_size + SEQ_LEN:]
axs[1].plot(test_dates, actual_prices, color='#1f77b4', label='Thực tế (Ground Truth)', alpha=0.8)
axs[1].plot(test_dates, predicted_prices, color='#ff7f0e', linestyle='--', label='Dự báo (Predicted)', linewidth=2)
axs[1].set_title(f'Kết quả trên tập Test (MSE: {mse_score:.2f})', fontsize=13)
axs[1].legend()
axs[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# =====================================================================
# BƯỚC 5: TRỰC QUAN HÓA MỨC TĂNG GIÁ ĐÓNG CỬA TRUNG BÌNH
# =====================================================================
print("5. Đang vẽ biểu đồ phân tích mức tăng giá đóng cửa...")

# 1. Mức tăng giá đóng cửa trung bình giữa các năm
yearly_avg = df.groupby(df.index.year)['Close'].mean()
yearly_increase = yearly_avg.diff().dropna()

# 2. Mức tăng giá theo từng tháng qua các năm (tính bằng chênh lệch giá ngày cuối dòng)
monthly_data = df.groupby([df.index.year, df.index.month])['Close'].last()
monthly_diff = monthly_data.diff().dropna()
avg_monthly_increase = monthly_diff.groupby(level=1).mean()

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('PHÂN TÍCH MỨC TĂNG/GIẢM GIÁ ĐÓNG CỬA', fontsize=16, fontweight='bold')

# Biểu đồ 1: Sự thay đổi qua các năm
colors_yr = ['#2ca02c' if val > 0 else '#d62728' for val in yearly_increase]
ax3.bar(yearly_increase.index, yearly_increase, color=colors_yr)
ax3.set_title('Mức tăng/giảm giá đóng cửa trung bình qua các năm', fontsize=13)
ax3.set_xlabel('Năm', fontsize=12)
ax3.set_ylabel('Mức thay đổi (USD)', fontsize=12)
if len(yearly_increase) > 0:
    ax3.set_xticks(yearly_increase.index)
ax3.grid(axis='y', alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Biểu đồ 2: Sự thay đổi theo tháng (Seasonality)
colors_mo = ['#2ca02c' if val > 0 else '#d62728' for val in avg_monthly_increase]
ax4.bar(avg_monthly_increase.index, avg_monthly_increase, color=colors_mo)
ax4.set_title('Mức tăng/giảm trung bình theo từng tháng', fontsize=13)
ax4.set_xlabel('Tháng', fontsize=12)
ax4.set_ylabel('Mức thay đổi (USD)', fontsize=12)
ax4.set_xticks(range(1, 13))
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("--- HOÀN THÀNH ---")