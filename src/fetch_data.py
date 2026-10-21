"""fetch_data.py
Data retrieval for Taiwan stock tickers using yfinance.
"""

from typing import Optional
import pandas as pd
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # Late import error will be raised if used without yfinance installed


def get_price_data(ticker: str) -> Optional[pd.DataFrame]:
    """Download roughly 5 years of daily data for a given Taiwan ticker.

    Args:
        ticker: e.g., "2330.TW" or "2454.TW" (Taiwan TWSE tickers in Yahoo format)

    Returns:
        A pandas DataFrame with the price data (index is dates, includes 'Close').
        Returns None if data couldn't be fetched or is empty.
    """
    if yf is None:
        print("Error: yfinance is not available. Install it to fetch data.")
        return None

    try:
        # period and interval cover roughly the last 5 years of daily data
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df is None or df.empty:
            print(f"Warning: No data returned for {ticker}")
            return None
        
        # Flatten MultiIndex columns if present (from recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Ensure we have a clean Close column for processing
        if 'Close' not in df.columns:
            print(f"Warning: 'Close' column missing for {ticker} data; cannot proceed.")
            return None
        df = df.dropna(subset=["Close"])  # remove rows with missing closing prices
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
