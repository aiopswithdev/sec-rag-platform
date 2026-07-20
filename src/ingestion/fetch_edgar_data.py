import os
from sec_edgar_downloader import Downloader

def download_financial_filings():
    # Define a clean directory to store raw data files
    target_dir = "./data"
    os.makedirs(target_dir, exist_ok=True)
    
    # CRITICAL: The SEC requires a declarative User-Agent header in this exact format.
    # Replace this with your data or a placeholder email to comply with sec.gov policies.
    user_agent = "aiopswithdev@gmail.com"
    
    dl = Downloader("ProductionRAGLab", user_agent, target_dir)
    
    # We will fetch the 10-K annual reports for the last 2 available years
    tickers = ["AAPL", "MSFT"]
    
    for ticker in tickers:
        print(f"Requesting 10-K filings for {ticker} from SEC EDGAR archive...")
        try:
            dl.get("10-K", ticker, after="2023-01-01")
            print(f"[-] Successfully downloaded {ticker} filings.")
        except Exception as e:
            print(f"[X] Failed to download data for {ticker}: {e}")

if __name__ == "__main__":
    download_financial_filings()
