📊 Dataflow Analysis

    Current Architecture

      1 ┌─────────────────────────────────────────────────────────────────────────┐
      2 │                        DATA SOURCES                                      │
      3 ├─────────────────────────────────────────────────────────────────────────┤
      4 │  tech_macro_aligned.csv    │  gdelt_sentiment_bq_aligned.csv           │
      5 │  (Market + Macro)          │  (Sentiment Scores per ticker)            │
      6 └──────────────┬─────────────┴──────────────────┬────────────────────────┘
      7                │                                │
      8                ▼                                ▼
      9 ┌─────────────────────────────────────────────────────────────────────────┐
     10 │                     LTSM FORECAST PIPELINE                               │
     11 │  (eval_rolling_forcast.py)                                              │
     12 │  ─────────────────────────────────────────────────                      │
     13 │  1. Load model weights (.pth)                                           │
     14 │  2. Build sequence window (SEQ_LEN=20)                                  │
     15 │  3. Run inference loop → predicted_prices                               │
     16 │  4. Output: MAE, RMSE, Visualization                                    │
     17 └─────────────────────────────────────────────────────────────────────────┘
     18                │
     19                │  OUTPUT: Predicted prices for Big 4 tickers
     20                ▼
     21 ┌─────────────────────────────────────────────────────────────────────────┐
     22 │                  STATISTICAL ANALYSIS PIPELINE                           │
     23 │  (statistical_analysis_big4.py)                                         │
     24 │  ─────────────────────────────────────────────────                      │
     25 │  PART 1: 95% CI for Portfolio Returns                                   │
     26 │           → Equal-weighted (25% each) Big 4 portfolio                   │
     27 │           → t-distribution based confidence interval                    │
     28 │                                                                         │
     29 │  PART 2: ANOVA on Sentiment Scores                                      │
     30 │           → H0: μ_AAPL = μ_MSFT = μ_GOOGL = μ_NVDA                      │
     31 │           → Post-hoc: Tukey's HSD if significant                        │
     32 └─────────────────────────────────────────────────────────────────────────┘
