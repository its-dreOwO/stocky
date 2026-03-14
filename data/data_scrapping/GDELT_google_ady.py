import os
import pandas as pd
from google.cloud import bigquery
import warnings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"
warnings.filterwarnings('ignore')

COMPANY_KEYWORDS = {
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet Inc',
    'NVDA': 'Nvidia',
    'META': 'Facebook|Meta Platforms'
}

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

OUTPUT_DIR = 'temp'
OUTPUT_FILE = 'gdelt_sentiment_bq_aligned.csv'

def query_gdelt_bigquery(client, ticker, keyword, start, end):
    print(f"[{ticker}] Initiating BigQuery execution for entity: {keyword}")
    
    query = f"""
        SELECT 
            CAST(PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS STRING) as Date,
            COUNT(*) as {ticker}_News_Volume,
            AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as {ticker}_Sentiment_Tone
        FROM 
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE 
            _PARTITIONTIME BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')
            AND REGEXP_CONTAINS(V2Organizations, r'(?i){keyword}')
        GROUP BY 
            Date
        ORDER BY 
            Date ASC
    """
    
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"  -> Extraction successful. Retrieved {len(df)} daily observations.")
            return df
        else:
            print(f"  -> [Warning] Zero observations returned for {ticker}.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  -> [Error] BigQuery execution failed: {repr(e)}")
        return pd.DataFrame()

def build_bigquery_dataset():
    print("Establishing connection to Google Cloud BigQuery...")
    client = bigquery.Client()
    master_df = pd.DataFrame()
    
    for ticker, keyword in COMPANY_KEYWORDS.items():
        df = query_gdelt_bigquery(client, ticker, keyword, START_DATE, END_DATE)
        
        if not df.empty:
            if master_df.empty:
                master_df = df
            else:
                master_df = pd.merge(master_df, df, on='Date', how='outer')
                
    if master_df.empty:
        return master_df

    print("\nApplying temporal interpolation and missing value imputation...")
    master_df = master_df.sort_values('Date').reset_index(drop=True)
    master_df = master_df.ffill().fillna(0) 
    
    return master_df

if __name__ == "__main__":
    dataset = build_bigquery_dataset()
    
    if not dataset.empty:
        print("\nBigQuery Extraction Complete. Data Sample:")
        print(dataset.head())
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        dataset.to_csv(output_path, index=False)
        print(f"\nSuccessfully serialized BigQuery sentiment proxy dataset to: {output_path}")
    else:
        print("\nPipeline failed. Dataset is empty. Verify GCP credentials and billing status.")
