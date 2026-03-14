import pandas as pd
import requests
import time
import os
from datetime import datetime

class SECScraper:
    def __init__(self, user_agent):
        self.headers = {'User-Agent': user_agent}
        self.base_url = "https://data.sec.gov/submissions/CIK{cik}.json"
        
        self.cik_map = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
            'NVDA': '0001045810',
            'META': '0001326801'
        }

    def get_filing_dates(self, ticker):
        cik = self.cik_map.get(ticker)
        if not cik:
            print(f"CIK for {ticker} not found.")
            return []

        url = self.base_url.format(cik=cik)
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                filings = data['filings']['recent']
                
                df = pd.DataFrame(filings)
                event_filings = df[df['form'].isin(['8-K', '10-Q', '10-K'])]
                
                return pd.to_datetime(event_filings['filingDate']).dt.date.tolist()
            else:
                print(f"Failed to fetch data for {ticker}: Status {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return []

if __name__ == "__main__":
    USER_AGENT = "Nguyen Hoai Khanh (K@itsdre.me)" 
    
    scraper = SECScraper(USER_AGENT)
    
    events = []
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']:
        print(f"Fetching event dates for {ticker}...")
        dates = scraper.get_filing_dates(ticker)
        for d in dates:
            events.append({'Ticker': ticker, 'Date': d, 'SEC_Event': 1})
        time.sleep(0.15) 

    if events:
        event_df = pd.DataFrame(events)
        output_dir = "../main_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "sec_events.csv")
        event_df.to_csv(output_path, index=False)
        print(f"SEC Event data saved to: {output_path}")
    else:
        print("No events were found.")
