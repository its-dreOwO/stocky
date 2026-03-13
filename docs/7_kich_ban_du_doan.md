# 📊 7 KỊCH BẢN AI MODEL DỰ ĐOÁN GIÁ CỔ PHIẾU

**Project:** Stocky - AI-Powered Stock Analysis  
**Model:** DLinear/LSTM Multivariate Forecast  
**Cập nhật:** 2025-03-12

---

## 🎯 TỔNG QUAN

### Mục Đích

Tài liệu này mô tả **7 kịch bản dự đoán** mà AI model có thể đưa ra khi phân tích và dự báo giá cổ phiếu của Big 4 Tech (AAPL, MSFT, GOOGL, NVDA).

### Quy Trình Dự Đoán

```
┌─────────────────────────────────────────────────────────────────────┐
│  QUY TRÌNH DỰ ĐOÁN 3 TRẠNG THÁI                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT: Sequence 192 ngày quá khứ (SEQ_LEN=192)                 │
│     [Day -192 → Day -1]                                            │
│                                                                     │
│  2. MODEL PREDICTION: Giá đóng cửa ngày mai                        │
│     Predicted_Price = Model(Sequence)                              │
│                                                                     │
│  3. SO SÁNH VỚI GIÁ HIỆN TẠI                                       │
│     Current_Price = Close_t                                        │
│     Price_Change = (Predicted - Current) / Current × 100%         │
│                                                                     │
│  4. PHÂN LOẠI XU HƯỚNG                                             │
│     • TĂNG    nếu Price_Change > +threshold                        │
│     • GIẢM   nếu Price_Change < -threshold                         │
│     • GIỮ NGUYÊN nếu -threshold ≤ Price_Change ≤ +threshold        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Ngưỡng Phân Loại

| Loại | Ngưỡng | Ký Hiệu |
|------|--------|---------|
| **Tăng mạnh** | > +2% | 🟢 STRONG_BUY |
| **Tăng nhẹ** | +0.5% → +2% | 🟡 BUY |
| **Đi ngang** | -0.5% → +0.5% | ⚪ HOLD |
| **Giảm nhẹ** | -2% → -0.5% | 🟠 SELL |
| **Giảm mạnh** | < -2% | 🔴 STRONG_SELL |

---

## 📈 KỊCH BẢN 1: TĂNG MẠNH (STRONG BULLISH)

### 🎯 Điều Kiện

```python
Price_Change > +2%
Predicted_Price > Current_Price × 1.02
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Higher Highs, Higher Lows | Xu hướng tăng rõ ràng |
| **RSI_14** | 55-65 | Động lượng dương, chưa quá mua |
| **ROC_5** | +3% đến +5% | Tốc độ thay đổi dương mạnh |
| **Sentiment** | +10 đến +30 | Tin tức tích cực |
| **SEC_Event** | 0 | Không có sự kiện tiêu cực |
| **Volume** | +20-30% vs MA20 | Khối lượng tăng |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $155.00                                         │
│  Price_Change:      +3.33%                                          │
│  → SIGNAL:          🟢 MUA MẠNH (STRONG BUY)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá trị | Diễn giải |
|----------|---------|-----------|
| **Mean Return** | +0.08%/ngày | Trung bình lịch sử |
| **Std Dev** | 1.25%/ngày | Độ biến động |
| **Z-Score** | +2.5σ | Ngoài 2.5 độ lệch chuẩn |
| **Confidence** | Cao | CI không overlap với 0 |

### 💡 Hành Động Khuyến Nghị

- **Action:** Mở vị thế mua ngay
- **Position Size:** 70-100% allocation
- **Stop Loss:** -3% từ entry
- **Take Profit:** +5-8%

### ⚠️ Rủi Ro

- Pullback sau tăng mạnh
- Profit taking từ short-term traders

---

## 📈 KỊCH BẢN 2: TĂNG NHẸ (WEAK BULLISH)

### 🎯 Điều Kiện

```python
+0.5% < Price_Change ≤ +2%
Current_Price × 1.005 < Predicted_Price ≤ Current_Price × 1.02
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Đi ngang có xu hướng tăng nhẹ | Consolidation với bias tăng |
| **RSI_14** | 45-55 | Trung tính |
| **ROC_5** | +0.5% đến +2% | Động lượng nhẹ |
| **Sentiment** | 0 đến +10 | Trung tính hơi tích cực |
| **Volume** | Bình thường | Không có đột biến |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $151.50                                         │
│  Price_Change:      +1.0%                                           │
│  → SIGNAL:          🟡 MUA (BUY/ACCUMULATE)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá trị | Diễn giải |
|----------|---------|-----------|
| **Z-Score** | +0.8σ | Trong 1 độ lệch chuẩn |
| **CI 95%** | [-0.15%, +0.32%] | Ngoài CI nhưng gần |
| **Confidence** | Trung bình | Cần thêm xác nhận |

### 💡 Hành Động Khuyến Nghị

- **Action:** Tích lũy từ từ
- **Position Size:** 30-50% allocation
- **Stop Loss:** -2% từ entry
- **Take Profit:** +3-4%

---

## 📉 KỊCH BẢN 3: GIẢM MẠNH (STRONG BEARISH)

### 🎯 Điều Kiện

```python
Price_Change < -2%
Predicted_Price < Current_Price × 0.98
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Lower Highs, Lower Lows | Xu hướng giảm rõ ràng |
| **RSI_14** | 30-40 | Động lượng âm, chưa quá bán |
| **ROC_5** | -3% đến -5% | Tốc độ thay đổi âm mạnh |
| **Sentiment** | -10 đến -30 | Tin tức tiêu cực |
| **SEC_Event** | 1 | Có sự kiện tiêu cực |
| **Volume** | +40-50% vs MA20 | Panic selling |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $145.00                                         │
│  Price_Change:      -3.33%                                          │
│  → SIGNAL:          🔴 BÁN MẠNH (STRONG SELL)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá trị | Diễn giải |
|----------|---------|-----------|
| **Z-Score** | -2.5σ | Ngoài 2.5 độ lệch chuẩn |
| **Mean Return** | +0.08% | Prediction trái ngược mean |
| **Confidence** | Cao | CI không overlap với 0 |

### 💡 Hành Động Khuyến Nghị

- **Action:** Giảm vị thế ngay
- **Position Size:** 0-20% allocation
- **Stop Loss:** +1% từ entry (nếu short)
- **Take Profit:** -5-8%

### ⚠️ Rủi Ro

- Short squeeze nếu quá bán
- Bounce kỹ thuật sau giảm mạnh

---

## 📉 KỊCH BẢN 4: GIẢM NHẸ (WEAK BEARISH)

### 🎯 Điều Kiện

```python
-2% ≤ Price_Change < -0.5%
Current_Price × 0.98 ≤ Predicted_Price < Current_Price × 0.995
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Đi ngang có xu hướng giảm nhẹ | Weak consolidation |
| **RSI_14** | 40-45 | Hơi âm |
| **ROC_5** | -0.5% đến -2% | Động lượng âm nhẹ |
| **Sentiment** | -10 đến 0 | Trung tính hơi tiêu cực |
| **Volume** | Hơi tăng | Không có panic |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $148.50                                         │
│  Price_Change:      -1.0%                                           │
│  → SIGNAL:          🟠 BÁN (SELL/REDUCE)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 💡 Hành Động Khuyến Nghị

- **Action:** Giảm tỷ trọng từ từ
- **Position Size:** 20-40% allocation
- **Stop Loss:** +1.5% từ entry
- **Take Profit:** -2-3%

---

## ➖ KỊCH BẢN 5: ĐI NGANG (SIDEWAYS/NEUTRAL)

### 🎯 Điều Kiện

```python
-0.5% ≤ Price_Change ≤ +0.5%
Current_Price × 0.995 ≤ Predicted_Price ≤ Current_Price × 1.005
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Đi ngang trong range hẹp | Consolidation |
| **RSI_14** | 48-52 | Hoàn toàn trung tính |
| **ROC_5** | -0.5% đến +0.5% | Không có động lực |
| **Sentiment** | -5 đến +5 | Không có tin nổi bật |
| **SEC_Event** | 0 | Không sự kiện |
| **Volume** | Thấp hơn MA20 | Thiếu động lực |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $150.30                                         │
│  Price_Change:      +0.2%                                           │
│  → SIGNAL:          ⚪ GIỮ (HOLD/WATCH)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá trị | Diễn giải |
|----------|---------|-----------|
| **Z-Score** | +0.1σ | Rất gần mean |
| **CI 95%** | [-0.15%, +0.32%] | Trong CI |
| **Significance** | p > 0.05 | Không có ý nghĩa thống kê |

### 💡 Hành Động Khuyến Nghị

- **Action:** Giữ nguyên vị thế
- **Position Size:** Maintain current
- **Strategy:** Wait for breakout signal
- **Note:** Không giao dịch trong range hẹp

---

## 🌪️ KỊCH BẢN 6: VOLATILITY CAO (HIGH UNCERTAINTY)

### 🎯 Điều Kiện

```python
Prediction CI width > 5%
Forecast Std Dev > 4%
```

### 📊 Input Features

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | Biến động mạnh ±3-5%/ngày | High volatility |
| **RSI_14** | Dao động nhanh 35→65 | Không ổn định |
| **ROC_5** | Biến động mạnh | Thiếu ổn định |
| **Sentiment** | Phân cực | Tin tốt và xấu xen kẽ |
| **SEC_Event** | 1 | Sự kiện quan trọng |
| **VIX/Treasury_10Y** | Tăng | Market stress |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $150.00                                         │
│  Predicted_Price:   $153.00                                         │
│  Prediction_CI_95%: [$145, $161]                                    │
│  CI Width:          10.7%                                           │
│  → SIGNAL:          ⚪ KHÔNG CHẮC CHẮN (UNCERTAIN)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá Trị | Bình Thường |
|----------|---------|-------------|
| **Forecast Std Dev** | 4-5% | 1.25% |
| **CI Width** | 10-12% | 2-3% |
| **Confidence** | Thấp | - |

### 💡 Hành Động Khuyến Nghị

- **Action:** Chờ đợi, quan sát
- **Position Size:** 0-20% allocation
- **Strategy:** Wait for clarity
- **Note:** Giảm position size trong môi trường uncertainty cao

---

## 🔄 KỊCH BẢN 7: ĐẢO CHIỀU (REVERSAL PATTERN)

### 🎯 Điều Kiện

```python
Current_Trend ≠ Predicted_Direction
Ví dụ: 5 ngày giảm liên tiếp nhưng model dự báo tăng
```

### 📊 Input Features (Reversal Tăng)

| Feature | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| **Xu hướng giá** | 5 ngày giảm liên tiếp | Extended downtrend |
| **RSI_14** | < 30 | Quá bán (Oversold) |
| **ROC_5** | -5% đến -8% | Động lượng âm cực độ |
| **Sentiment** | -30 đến -50 | Tiêu cực cực độ |
| **Volume** | Rất cao | Capitulation volume |
| **Pattern** | Bottom formation | Potential reversal |

### 🤖 Model Output

```
┌─────────────────────────────────────────────────────────────────────┐
│  Current_Price:     $145.00 (sau 5 ngày giảm)                       │
│  Predicted_Price:   $150.00                                         │
│  Price_Change:      +3.45%                                          │
│  → SIGNAL:          🟢 ĐẢO CHIỀU TĂNG (CONTRARIAN BUY)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 📐 Statistical Context

| Thống kê | Giá trị | Diễn giải |
|----------|---------|-----------|
| **Mean Reversion** | Strong | Giá có xu hướng quay về mean |
| **Skewness** | Âm mạnh | Khả năng rebound cao |
| **Percentile** | 5th percentile | Giá rẻ bất thường |

### 💡 Hành Động Khuyến Nghị

- **Action:** Mua ngược xu hướng
- **Position Size:** 40-60% allocation
- **Stop Loss:** -4% từ entry
- **Take Profit:** +6-10%

### ⚠️ Rủi Ro

- "Bắt dao rơi" - Trend tiếp tục giảm
- Cần xác nhận thêm từ volume và price action

---

## 📊 BẢNG TÓM TẮT 7 KỊCH BẢN

| # | Kịch bản | Điều kiện | Tín hiệu | Action | Position |
|---|----------|-----------|----------|--------|----------|
| 1 | 🟢 Tăng mạnh | > +2% | STRONG_BUY | Mua ngay | 70-100% |
| 2 | 🟡 Tăng nhẹ | +0.5% → +2% | BUY | Tích lũy | 30-50% |
| 3 | 🔴 Giảm mạnh | < -2% | STRONG_SELL | Bán ngay | 0-20% |
| 4 | 🟠 Giảm nhẹ | -2% → -0.5% | SELL | Giảm tỷ trọng | 20-40% |
| 5 | ⚪ Đi ngang | -0.5% → +0.5% | HOLD | Giữ/Quan sát | Maintain |
| 6 | ⚪ Volatility cao | CI rộng | UNCERTAIN | Chờ đợi | 0-20% |
| 7 | 🟢 Đảo chiều | Ngược trend | CONTRARIAN | Mua ngược | 40-60% |

---

## 🔧 IMPLEMENTATION

### Code Mẫu

```python
# File: practice/prediction_scenarios.py

from enum import Enum

class Signal(Enum):
    STRONG_BUY = "MUA MẠNH"
    BUY = "MUA"
    HOLD = "GIỮ"
    SELL = "BÁN"
    STRONG_SELL = "BÁN MẠNH"
    UNCERTAIN = "KHÔNG CHẮC CHẮN"

def classify_prediction(current_price, predicted_price, 
                        threshold_strong=0.02, threshold_weak=0.005):
    """
    Phân loại tín hiệu dựa trên % thay đổi giá
    """
    price_change = (predicted_price - current_price) / current_price
    price_change_pct = price_change * 100
    
    if price_change > threshold_strong:
        return Signal.STRONG_BUY, price_change_pct, "TĂNG MẠNH"
    elif price_change > threshold_weak:
        return Signal.BUY, price_change_pct, "TĂNG NHẸ"
    elif price_change < -threshold_strong:
        return Signal.STRONG_SELL, price_change_pct, "GIẢM MẠNH"
    elif price_change < -threshold_weak:
        return Signal.SELL, price_change_pct, "GIẢM NHẸ"
    else:
        return Signal.HOLD, price_change_pct, "ĐI NGANG"
```

---

## 📌 LƯU Ý QUAN TRỌNG

### ⚠️ Cảnh Báo Rủi Ro

1. **Model không chính xác 100%** - Luôn sử dụng stop loss
2. **Overfitting** - Model có thể hoạt động tốt trên historical data nhưng kém trên real-time
3. **Market Regime Change** - Model cần retrain khi market structure thay đổi
4. **Black Swan Events** - Các sự kiện không lường trước được

### ✅ Best Practices

1. **Kết hợp nhiều tín hiệu** - Không dựa chỉ vào 1 model
2. **Quản lý rủi ro** - Position sizing phù hợp
3. **Backtesting** - Kiểm tra trên historical data
4. **Paper trading** - Test trước khi dùng tiền thật

---

## 📚 TÀI LIỆU THAM KHẢO

- `AI/LTSM/main.py` - Model architecture
- `AI/LTSM/eval_rolling_forcast.py` - Rolling forecast evaluation
- `practice/statistical_analysis_big4_enhanced.py` - Statistical analysis
- `visualization/big4_unified_visualization.py` - Visualization

---

**Version:** 1.0  
**Author:** Stocky Team  
**Last Updated:** 2025-03-12
