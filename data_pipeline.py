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
    
    # Convert string dates to datetime objects
    try:
        since_dt = pd.Timestamp(since) if isinstance(since, str) else since
        until_dt = pd.Timestamp(until) if isinstance(until, str) else until
        
        # Ensure until is after since
        if until_dt <= since_dt:
            raise ValueError(f"End date ({until_dt}) must be after start date ({since_dt})")
            
        # For intraday data, limit the date range to last 60 days
        if interval in ['1m', '5m', '15m', '30m']:
            max_days_ago = pd.Timestamp.now() - pd.Timedelta(days=59)  # 59 days to be safe
            if since_dt < max_days_ago:
                print(f"Warning: {interval} data is limited to last 60 days. Adjusting date range...")
                since_dt = max_days_ago
                since = since_dt.strftime('%Y-%m-%d')
                
            # Cap the end date to now
            now = pd.Timestamp.now()
            if until_dt > now:
                until_dt = now
                until = until_dt.strftime('%Y-%m-%d')
    except Exception as e:
        raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}")
    
    # Format symbol for yfinance
    yf_symbol = str(symbol).upper()
    
    # Handle different symbol formats
    if '/' in yf_symbol:
        # Handle pair format (e.g., 'BTC/USDT' -> 'BTC-USD')
        base, quote = yf_symbol.split('/')
        yf_symbol = f"{base}-{quote}"
    
    # Common replacements
    yf_symbol = yf_symbol.replace('USDT', 'USD')
    yf_symbol = yf_symbol.replace('--', '-')
    
    retries = max_retries
    df = pd.DataFrame()
    last_error = None
    
    for attempt in range(retries):
        try:
            print(f"Fetching {yf_symbol} data from {since} to {until}...")
            
            # Try different symbol variations
            symbol_variations = [
                yf_symbol,  # Try as-is first
                f"{yf_symbol}-USD" if not yf_symbol.endswith('-USD') else None,  # Add -USD if not present
                f"{yf_symbol}.NS",  # Try NSE (India) exchange
                yf_symbol.replace('-USD', '.NS') if yf_symbol.endswith('-USD') else None,  # Replace -USD with .NS
                yf_symbol.split('-')[0]  # Try just the base symbol (e.g., 'BTC' instead of 'BTC-USD')
            ]
            
            for sym in filter(None, symbol_variations):
                if sym == yf_symbol:
                    print(f"Trying {sym}...")
                else:
                    print(f"No data, trying {sym}...")
                    
                ticker = yf.Ticker(sym)
                df = ticker.history(
                    start=since_dt,
                    end=until_dt,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if not df.empty:
                    print(f"Successfully fetched data for {sym}")
                    yf_symbol = sym  # Update the symbol to the one that worked
                    break
                
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
    
    # Reset index to make DatetimeIndex a column
    df = df.reset_index()
    
    # Standardize column names (case-insensitive matching)
    column_mapping = {
        'index': 'timestamp',  # This captures the DatetimeIndex when reset
        'date': 'timestamp',
        'Date': 'timestamp',
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume',
        'Dividends': 'dividends',
        'Stock Splits': 'splits'
    }
    
    # Rename columns using case-insensitive matching
    df_renamed = pd.DataFrame()
    for col in df.columns:
        lower_col = str(col).lower()
        if lower_col in column_mapping:
            df_renamed[column_mapping[lower_col]] = df[col]
        elif col in column_mapping:
            df_renamed[column_mapping[col]] = df[col]
        else:
            df_renamed[col] = df[col]
    
    df = df_renamed
    
    # Ensure timestamp column exists and is datetime type
    if 'timestamp' not in df.columns and 'datetime' in df.columns:
        df['timestamp'] = df['datetime']
    
    if 'timestamp' not in df.columns:
        raise ValueError("Could not find timestamp/datetime column in the downloaded data")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Localize to UTC if no timezone info
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Ensure we have the required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            if col == 'volume':
                df['volume'] = 0  # Default to 0 if volume not available
            else:
                raise ValueError(f"Required column '{col}' not found in the downloaded data")
    
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
    min_required_points = {
        '1m': 1000,  # 1-minute data should have many points
        '5m': 200,
        '15m': 100,
        '30m': 50,
        '1h': 24,
        '1d': 5,    # For testing with small date ranges
        '1w': 2,
        '1M': 1
    }
    
    min_points = min_required_points.get(timeframe, 5)  # Default to 5 if timeframe not found
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Please check the symbol and date range.")
    
    if len(df) < min_points:
        print(f"Warning: Only {len(df)} data points found (minimum recommended: {min_points})")
        # Don't fail for small datasets - just warn
        # This allows testing with small date ranges
    
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
