import os
import time
from typing import Tuple
import pandas as pd
import yfinance as yf


def download_ohlcv_yfinance(
    symbol: str,
    timeframe: str,
    since: str,
    until: str,
    save_path: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download OHLCV data using yfinance with proper date range handling."""
    # Map timeframe to yfinance format
    tf_mapping = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '1d': '1d',
        '1w': '1wk',
        '1M': '1mo'
    }
    
    interval = tf_mapping.get(timeframe, '1d')
    
    # For intraday data, limit the date range to last 60 days
    since_dt = pd.Timestamp(since)
    until_dt = pd.Timestamp(until)
    
    if interval in ['1m', '5m', '15m', '30m']:
        max_days_ago = pd.Timestamp.now() - pd.Timedelta(days=59)  # 59 days to be safe
        if since_dt < max_days_ago:
            print(f"Warning: {interval} data is limited to last 60 days. Adjusting date range...")
            since_dt = max_days_ago
            since = since_dt.strftime('%Y-%m-%d')
            
        # Also cap the end date to today
        if until_dt > pd.Timestamp.now():
            until_dt = pd.Timestamp.now()
            until = until_dt.strftime('%Y-%m-%d')
    
    # Format symbol for yfinance (e.g., 'BTC/USDT' -> 'BTC-USD')
    yf_symbol = symbol.upper().replace('/', '-').replace('USDT', '-USD')
    # Remove any double dashes that might have been created
    yf_symbol = yf_symbol.replace('--', '-')
    
    retries = max_retries
    df = pd.DataFrame()
    last_error = None
    
    for attempt in range(retries):
        try:
            # First try with the main symbol
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=since,
                end=until,
                interval=interval,
                auto_adjust=True
            )
            
            # If no data, try with .NS (NSE) suffix for Indian stocks
            if df.empty:
                print(f"No data found for {yf_symbol}, trying with .NS (NSE) suffix...")
                ticker = yf.Ticker(f"{yf_symbol}.NS")
                df = ticker.history(
                    start=since,
                    end=until,
                    interval=interval,
                    auto_adjust=True
                )
            
            # If still no data, try with -USD suffix
            if df.empty and '-USD' not in yf_symbol:
                print(f"Trying with -USD suffix...")
                ticker = yf.Ticker(f"{yf_symbol}-USD")
                df = ticker.history(
                    start=since,
                    end=until,
                    interval=interval,
                    auto_adjust=True
                )
                
            # If we have data, break the retry loop
            if not df.empty:
                break
                
            
        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempt + 1}/{retries} failed: {last_error}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                error_msg = (
                    f"Failed to download data after {retries} attempts.\n"
                    f"Symbol: {yf_symbol}\n"
                    f"Timeframe: {timeframe}\n"
                    f"Date range: {since} to {until}\n"
                    f"Last error: {last_error}\n\n"
                    "Common solutions:\n"
                    "1. Check your internet connection\n"
                    "2. Verify the symbol is correct (e.g., 'AAPL', 'MSFT', 'BTC-USD')\n"
                    "3. Try a different symbol or check if the market is open\n"
                    "4. Try a different date range"
                )
                print(error_msg)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    if df.empty:
        print(f"No data found for {yf_symbol} with interval {interval}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Reset index and rename columns to match expected format
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Ensure timestamp is timezone-aware
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Filter by date range to be safe
    since_dt = pd.to_datetime(since, utc=True)
    until_dt = pd.to_datetime(until, utc=True)
    df = df[(df['timestamp'] >= since_dt) & (df['timestamp'] <= until_dt)]
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_or_download_ohlcv(
    path: str,
    exchange_id: str,  # Kept for backward compatibility, not used with yfinance
    symbol: str,
    timeframe: str,
    since: str,
    until: str,
) -> pd.DataFrame:
    # First try to load existing data
    if os.path.exists(path):
        df = load_ohlcv_csv(path)
        if not df.empty:
            return df
    
    # If no valid data exists, try to download it
    df = download_ohlcv_yfinance(symbol, timeframe, since, until, path)
    
    # Verify we have valid data
    if df.empty or len(df) < 100:  # Require at least 100 data points
        raise ValueError(f"Insufficient data for {symbol}. Please check your internet connection and symbol validity.")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp'")
    train_end_ts = pd.to_datetime(train_end, utc=True)
    test_start_ts = pd.to_datetime(test_start, utc=True)
    train_df = df[df["timestamp"] <= train_end_ts].copy()
    test_df = df[df["timestamp"] >= test_start_ts].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
