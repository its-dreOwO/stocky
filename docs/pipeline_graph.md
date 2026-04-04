# Project Pipeline Graph

```mermaid
graph TD
    %% Data Sources
    subgraph "Data Acquisition & Ingestion"
        A1[GDELT BigQuery] --> B1[gdelt_sentiment_bq_aligned.csv]
        A2[SEC Website] -->|sec_scraper.py| B2[sec_events.csv]
        A3[Market/Macro APIs] --> B3[tech_macro_aligned.csv]
    end

    %% Preprocessing
    subgraph "Preprocessing & Feature Engineering"
        B1 & B2 & B3 --> C1[dataset_builder.py]
        C1 -->|MultivariateStockDataset| C2[Training/Testing Sets]
        C1 -->|Feature Weights| C2
        C1 -->|ROC, RSI, Sentiments, SEC Events| C2
    end

    %% Modeling
    subgraph "Modeling & Training"
        C2 --> D1[AI/LTSM/main.py]
        C2 --> D2[ARIMA/main.py]
        C2 --> D3[Logistic Regression/main.py]
        D1 -->|DLinear, NLinear, LSTM, GRU| E1[Trained Models (.pth)]
        D4[hyperparameter_tuner.py] -.-> D1
    end

    %% Evaluation
    subgraph "Evaluation & Inference"
        E1 --> F1[eval_rolling_forcast.py]
        F1 -->|Metrics: MAE, RMSE| G1[Final Performance Report]
        F1 -->|Predicted Prices| G2[Rolling Forecast Graphs]
    end

    %% Visualization
    subgraph "Visualization & Reporting"
        G2 --> H1[weighted_parameters_heatmap.py]
        G2 --> H2[box_plots_static.py]
        G2 --> H3[ohlc_2025_line_graph.py]
        H1 & H2 & H3 --> I1[visualization/output/]
        I1 --> J1[Power BI: Stocks_visual.pbix]
    end
```

## Description

1.  **Data Acquisition**: Sentiment data is pulled from GDELT, SEC events are scraped, and market/macro data is aligned.
2.  **Preprocessing**: Features like ROC and RSI are engineered, and data is structured into sequences for multivariate forecasting.
3.  **Modeling**: The core pipeline supports deep learning models (DLinear, NLinear, LSTM, GRU) as well as traditional statistical models (ARIMA) and basic machine learning (Logistic Regression).
4.  **Evaluation**: Models are evaluated using a rolling forecast approach, providing metrics like MAE and RMSE.
5.  **Visualization**: Results are visualized through heatmaps, box plots, and line graphs, ultimately feeding into a Power BI dashboard.
