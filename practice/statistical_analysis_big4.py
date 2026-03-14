import pandas as pd
import numpy as np
from scipy import stats

df_stocks = pd.read_csv(r'data\main_data\tech_macro_aligned.csv')

df_sent = pd.read_csv(r'data\data_scrapping\temp\gdelt_sentiment_bq_aligned.csv')

print("=" * 80)
print("PHẦN 1: KHOẢNG TIN CẬY 95% CHO TỶ SUẤT SINH LỜI TRUNG BÌNH HÀNG NGÀY")
print("Danh mục: Big 4 Tech (AAPL, MSFT, GOOGL, NVDA) - đều trọng số 25%")
print("=" * 80)

big4_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
df_big4 = df_stocks[df_stocks['Ticker'].isin(big4_tickers)].copy()

df_close = df_big4.pivot_table(index='Date', columns='Ticker', values='Close')

df_returns = df_close.pct_change().dropna()

weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_returns = (df_returns * weights).sum(axis=1)

n = len(portfolio_returns)
mean_return = portfolio_returns.mean()
std_return = portfolio_returns.std(ddof=1)  

alpha = 0.05
df = n - 1  
t_critical = stats.t.ppf(1 - alpha/2, df)

margin_of_error = t_critical * (std_return / np.sqrt(n))

ci_lower = mean_return - margin_of_error
ci_upper = mean_return + margin_of_error

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

print("\n" + "=" * 80)
print("PHẦN 2: KIỂM ĐỊNH SỰ KHÁC BIỆT VỀ CẢM XÚC TRUYỀN THÔNG (SENTIMENT SCORE)")
print("Giữa 4 mã cổ phiếu: AAPL, MSFT, GOOGL, NVDA")
print("=" * 80)

sentiment_cols = {
    'AAPL': 'AAPL_Sentiment_Tone',
    'MSFT': 'MSFT_Sentiment_Tone',
    'GOOGL': 'GOOGL_Sentiment_Tone',
    'NVDA': 'NVDA_Sentiment_Tone'
}

df_sent_big4 = pd.DataFrame()
for ticker, col in sentiment_cols.items():
    df_sent_big4[ticker] = df_sent[col]

df_sent_clean = df_sent_big4.dropna()

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

print("\n--- KIỂM ĐỊNH ONE-WAY ANOVA ---")

f_statistic, p_value = stats.f_oneway(
    df_sent_clean['AAPL'],
    df_sent_clean['MSFT'],
    df_sent_clean['GOOGL'],
    df_sent_clean['NVDA']
)

k = 4  
N = len(df_sent_clean) * k  
df_between = k - 1  
df_within = len(df_sent_clean) * k - k  

alpha = 0.05
f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

print(f"\nMức ý nghĩa (α): {alpha}")
print(f"\nKết quả ANOVA:")
print(f"  F-statistic: {f_statistic:.6f}")
print(f"  F-critical (α={alpha}): {f_critical:.6f}")
print(f"  p-value: {p_value:.10f}")
print(f"  Bậc tự do: df_between = {df_between}, df_within = {df_within}")

print("\n--- KẾT LUẬN ---")
if p_value < alpha:
    print(f"Vì p-value ({p_value:.10f}) < α ({alpha})")
    print("→ BÁC BỎ H₀")
    print("→ Có sự khác biệt có ý nghĩa thống kê về Sentiment Score giữa ít nhất 2 mã cổ phiếu.")
else:
    print(f"Vì p-value ({p_value:.10f}) ≥ α ({alpha})")
    print("→ KHÔNG ĐỦ CƠ SỞ BÁC BỎ H₀")
    print("→ Không có sự khác biệt có ý nghĩa thống kê về Sentiment Score giữa 4 mã cổ phiếu.")

if p_value < alpha:
    print("\n--- PHÂN TÍCH HẬU KIỂM (TUKEY'S HSD) ---")
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    all_data = []
    all_groups = []
    for ticker in big4_tickers:
        all_data.extend(df_sent_clean[ticker].tolist())
        all_groups.extend([ticker] * len(df_sent_clean[ticker]))
    
    tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
    print("\nKết quả Tukey's HSD (α = 0.05):")
    print(tukey_result)

print("\n" + "=" * 80)
print("KẾT THÚC PHÂN TÍCH")
print("=" * 80)
