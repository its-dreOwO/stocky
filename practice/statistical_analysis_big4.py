"""
Phân tích Thống kê cho Danh mục "Big 4 Tech" (AAPL, MSFT, GOOGL, NVDA)
1. Ước lượng khoảng tin cậy 95% cho Tỷ suất sinh lời trung bình hàng ngày
2. Kiểm định sự khác biệt về Sentiment Score giữa 4 mã
"""

import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# ĐỌC DỮ LIỆU
# =============================================================================

# Đọc dữ liệu giá
df_stocks = pd.read_csv(r'data\main_data\tech_macro_aligned.csv')

# Đọc dữ liệu sentiment
df_sent = pd.read_csv(r'data\data_scrapping\temp\gdelt_sentiment_bq_aligned.csv')

# =============================================================================
# PHẦN 1: KHOẢNG TIN CẬY 95% CHO TỶ SUẤT SINH LỜI TRUNG BÌNH HÀNG NGÀY
# =============================================================================

print("=" * 80)
print("PHẦN 1: KHOẢNG TIN CẬY 95% CHO TỶ SUẤT SINH LỜI TRUNG BÌNH HÀNG NGÀY")
print("Danh mục: Big 4 Tech (AAPL, MSFT, GOOGL, NVDA) - đều trọng số 25%")
print("=" * 80)

# Lọc dữ liệu cho Big 4
big4_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
df_big4 = df_stocks[df_stocks['Ticker'].isin(big4_tickers)].copy()

# Pivot để có dữ liệu giá Close cho mỗi mã
df_close = df_big4.pivot_table(index='Date', columns='Ticker', values='Close')

# Tính daily returns cho mỗi mã
df_returns = df_close.pct_change().dropna()

# Tạo portfolio đều trọng số (25% mỗi mã)
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_returns = (df_returns * weights).sum(axis=1)

# Tính các thống kê
n = len(portfolio_returns)
mean_return = portfolio_returns.mean()
std_return = portfolio_returns.std(ddof=1)  # sample standard deviation

# Tính critical value (t-value) cho 95% CI
alpha = 0.05
df = n - 1  # degrees of freedom
t_critical = stats.t.ppf(1 - alpha/2, df)

# Biên độ sai số (Margin of Error)
margin_of_error = t_critical * (std_return / np.sqrt(n))

# Khoảng tin cậy 95%
ci_lower = mean_return - margin_of_error
ci_upper = mean_return + margin_of_error

# Chuyển đổi sang phần trăm để dễ đọc
mean_return_pct = mean_return * 100
std_return_pct = std_return * 100
margin_of_error_pct = margin_of_error * 100
ci_lower_pct = ci_lower * 100
ci_upper_pct = ci_upper * 100

print(f"\n{'Thống kê':<35} | {'Giá trị':<20}")
print("-" * 60)
print(f"{'Cỡ mẫu (n)':<35} | {n:<20,}")
print(f"{'Trung bình mẫu (Mean)':<35} | {mean_return_pct:<20.6f}%")
print(f"{'Độ lệch chuẩn (Std Dev)':<35} | {std_return_pct:<20.6f}%")
print(f"{'Bậc tự do (df)':<35} | {df:<20,}")
print(f"{'Critical value (t)':<35} | {t_critical:<20.6f}")
print(f"{'Biên độ sai số (E)':<35} | {margin_of_error_pct:<20.6f}%")
print(f"{'CI 95% - Cận dưới':<35} | {ci_lower_pct:<20.6f}%")
print(f"{'CI 95% - Cận trên':<35} | {ci_upper_pct:<20.6f}%")
print("-" * 60)
print(f"\n=> Khoảng tin cậy 95%: [{ci_lower_pct:.6f}%, {ci_upper_pct:.6f}%]")
print(f"   Hoặc: {mean_return_pct:.6f}% ± {margin_of_error_pct:.6f}%")

# =============================================================================
# PHẦN 2: KIỂM ĐỊNH SỰ KHÁC BIỆT VỀ SENTIMENT SCORE
# =============================================================================

print("\n" + "=" * 80)
print("PHẦN 2: KIỂM ĐỊNH SỰ KHÁC BIỆT VỀ CẢM XÚC TRUYỀN THÔNG (SENTIMENT SCORE)")
print("Giữa 4 mã cổ phiếu: AAPL, MSFT, GOOGL, NVDA")
print("=" * 80)

# Chuẩn bị dữ liệu sentiment cho Big 4
sentiment_cols = {
    'AAPL': 'AAPL_Sentiment_Tone',
    'MSFT': 'MSFT_Sentiment_Tone',
    'GOOGL': 'GOOGL_Sentiment_Tone',
    'NVDA': 'NVDA_Sentiment_Tone'
}

# Tạo dataframe với sentiment scores cho từng mã
df_sent_big4 = pd.DataFrame()
for ticker, col in sentiment_cols.items():
    df_sent_big4[ticker] = df_sent[col]

# Loại bỏ các giá trị NaN
df_sent_clean = df_sent_big4.dropna()

# Thống kê mô tả cho từng mã
print("\n--- THỐNG KÊ MÔ TẢ SENTIMENT SCORE ---")
print(f"\n{'Mã':<10} | {'N':<10} | {'Trung bình':<15} | {'Độ lệch chuẩn':<15}")
print("-" * 60)

sentiment_data = []
for ticker in big4_tickers:
    data = df_sent_clean[ticker]
    n_sent = len(data)
    mean_sent = data.mean()
    std_sent = data.std(ddof=1)
    sentiment_data.append(data)
    print(f"{ticker:<10} | {n_sent:<10,} | {mean_sent:<15.6f} | {std_sent:<15.6f}")

print("-" * 60)

# Thực hiện One-Way ANOVA
print("\n--- KIỂM ĐỊNH ONE-WAY ANOVA ---")

# Giả thuyết:
# H0: μ_AAPL = μ_MSFT = μ_GOOGL = μ_NVDA (không có sự khác biệt)
# H1: Ít nhất một giá trị trung bình khác biệt

# Thực hiện ANOVA
f_statistic, p_value = stats.f_oneway(
    df_sent_clean['AAPL'],
    df_sent_clean['MSFT'],
    df_sent_clean['GOOGL'],
    df_sent_clean['NVDA']
)

# Các bậc tự do
k = 4  # số nhóm
N = len(df_sent_clean) * k  # tổng số quan sát
df_between = k - 1  # bậc tự do giữa các nhóm
df_within = len(df_sent_clean) * k - k  # bậc tự do trong các nhóm

alpha = 0.05
f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

print(f"\nMức ý nghĩa (α): {alpha}")
print(f"\nKết quả ANOVA:")
print(f"  F-statistic: {f_statistic:.6f}")
print(f"  F-critical (α={alpha}): {f_critical:.6f}")
print(f"  p-value: {p_value:.10f}")
print(f"  Bậc tự do: df_between = {df_between}, df_within = {df_within}")

# Quyết định
print("\n--- KẾT LUẬN ---")
if p_value < alpha:
    print(f"Vì p-value ({p_value:.10f}) < α ({alpha})")
    print("→ BÁC BỎ H₀")
    print("→ Có sự khác biệt có ý nghĩa thống kê về Sentiment Score giữa ít nhất 2 mã cổ phiếu.")
else:
    print(f"Vì p-value ({p_value:.10f}) ≥ α ({alpha})")
    print("→ KHÔNG ĐỦ CƠ SỞ BÁC BỎ H₀")
    print("→ Không có sự khác biệt có ý nghĩa thống kê về Sentiment Score giữa 4 mã cổ phiếu.")

# =============================================================================
# PHÂN TÍCH HẬU KIỂM (POST-HOC) NẾU CÓ Ý NGHĨA
# =============================================================================

if p_value < alpha:
    print("\n--- PHÂN TÍCH HẬU KIỂM (TUKEY'S HSD) ---")
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Chuẩn bị dữ liệu cho Tukey's HSD
    all_data = []
    all_groups = []
    for ticker in big4_tickers:
        all_data.extend(df_sent_clean[ticker].tolist())
        all_groups.extend([ticker] * len(df_sent_clean[ticker]))
    
    # Thực hiện Tukey's HSD
    tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
    print("\nKết quả Tukey's HSD (α = 0.05):")
    print(tukey_result)

print("\n" + "=" * 80)
print("KẾT THÚC PHÂN TÍCH")
print("=" * 80)
