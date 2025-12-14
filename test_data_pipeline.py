"""Test script for the data pipeline functionality."""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from data_pipeline import get_or_download_ohlcv, download_ohlcv_yfinance

def test_symbol(symbol, name, timeframe='1d', start_date=None, end_date=None):
    """Test data download for a given symbol and timeframe."""
    if start_date is None:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)  # Default to last 30 days
    
    print(f"\n{'='*50}")
    print(f"Testing {name} ({symbol}) - {timeframe} timeframe")
    print(f"{'='*50}")
    
    # Calculate days for date range validation
    days = (end_date - start_date).days
    
    path = f"test_{symbol.replace('/', '_')}_{timeframe}.csv"
    
    try:
        # Test get_or_download_ohlcv
        print(f"\n[1/2] Testing get_or_download_ohlcv...")
        df = get_or_download_ohlcv(
            path=path,
            exchange_id='yfinance',
            symbol=symbol,
            timeframe=timeframe,
            since=start_date.strftime('%Y-%m-%d'),
            until=end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            print("❌ No data returned")
            return False
            
        print("✅ Data retrieved successfully")
        print(f"    - Rows: {len(df)}")
        print(f"    - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"    - Columns: {', '.join(df.columns)}")
        
        # Basic data validation
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"❌ Missing required columns: {', '.join(missing)}")
            return False
            
        # Test date range (only if we have multiple days of data)
        if len(df) > 1:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            expected_min_days = max(1, int(days * 0.8))  # At least 1 day, allow 20% tolerance
            if date_range < expected_min_days:
                print(f"⚠️  Date range ({date_range} days) shorter than expected ({expected_min_days} days)")
        else:
            print(f"ℹ️  Single day of data returned")
        
        # Test direct download function
        print(f"\n[2/2] Testing direct download...")
        df_direct = download_ohlcv_yfinance(
            symbol=symbol,
            timeframe=timeframe,
            since=start_date.strftime('%Y-%m-%d'),
            until=end_date.strftime('%Y-%m-%d'),
            save_path=path.replace('.csv', '_direct.csv')
        )
        
        if df_direct.empty:
            print("❌ Direct download failed")
            return False
            
        print("✅ Direct download successful")
        print(f"    - Rows: {len(df_direct)}")
        print(f"    - Date range: {df_direct['timestamp'].min()} to {df_direct['timestamp'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    # Use recent dates that are likely to have data
    end_date = datetime.now() - timedelta(days=7)  # One week ago
    
    # Test cases with realistic date ranges
    test_cases = [
        # (symbol, name, timeframe, days, start_date_offset)
        ('AAPL', 'Apple Stock', '1d', 30, 37),  # 30 days ending 1 week ago
        ('BTC-USD', 'Bitcoin', '1d', 30, 37),   # 30 days ending 1 week ago
        ('MSFT', 'Microsoft Stock', '1h', 2, 9),  # 2 days of hourly data
        ('GOOGL', 'Alphabet Stock', '1d', 30, 37),
        ('SPY', 'S&P 500 ETF', '1d', 30, 37),
        ('QQQ', 'Nasdaq 100 ETF', '1d', 30, 37),
    ]
    
    # Update test dates to use the fixed end date
    test_cases = [
        (symbol, name, tf, days, end_date - timedelta(days=offset))
        for symbol, name, tf, days, offset in test_cases
    ]
    
    print("Starting data pipeline tests...\n")
    results = {}
    
    for symbol, name, tf, days, test_end_date in test_cases:
        start_date = test_end_date - timedelta(days=days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = test_end_date.strftime('%Y-%m-%d')
        test_name = f"{name} ({symbol}, {tf})"
        print(f"\n{'='*30} {test_name} {'='*30}")
        success = test_symbol(symbol, f"{name} ({start_str} to {end_str})", tf, start_date, test_end_date)
        results[test_name] = '✅ PASSED' if success else '❌ FAILED'
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    for test, result in results.items():
        print(f"{result} - {test}")
    
    # Cleanup test files
    for f in os.listdir('.'):
        if f.startswith('test_') and f.endswith('.csv'):
            os.remove(f)

if __name__ == "__main__":
    run_tests()
