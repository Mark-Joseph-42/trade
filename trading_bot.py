import asyncio
import time
import sys
import logging
import pandas as pd
import numpy as np
import json
import os
import ccxt.async_support as ccxt
from collections import deque
from rich.console import Console
from rich.table import Table

# --- Configuration Import ---
try:
    from config import (
        EXCHANGE_ID, API_KEY, API_SECRET, API_PASSWORD,
        SIMULATION_MODE, SIM_STARTING_BALANCE, INVEST_FRACTION, MAX_ACTIVE_TRADES,
        FEE_RATE, UNIVERSE_SIZE, RISK_REWARD_RATIO, STOP_LOSS_PCT, BUY_SCORE_THRESHOLD,
        KLINE_INTERVAL, KLINE_HISTORY_CANDLES, MIN_24H_VOLUME_USDT, 
        ANALYSIS_INTERVAL_SECONDS, MAX_TRADE_DURATION_CANDLES, 
        PERFORMANCE_LOG_FILE, SELF_LEARNING_ENABLED, DYNAMIC_POSITION_SIZING,
        VOLATILITY_LOOKBACK, ATR_MULTIPLIER_SL, 
        ENABLE_MACD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
        ENABLE_MAE, MAE_LENGTH, MAE_PCT_OFFSET,
        ENABLE_ICT
    )
except ImportError as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('bot_debug.log', mode='w')]
)
logger = logging.getLogger(__name__)
console = Console()

# --- Global State ---
state = {} 
monitor_tasks = {}
sim_balance = SIM_STARTING_BALANCE
active_trades = []
status_log = deque(maxlen=20)

# Store 24 hours of minute-by-minute data (1440 minutes)
performance_history = deque(maxlen=1440) 
last_history_update = 0

start_time = time.time()
strategy_performance = {}

# --- Helpers ---
def update_status(message: str):
    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    if not status_log or status_log[-1] != log_msg:
        status_log.append(log_msg)
    logger.info(message)

def compute_final_balance():
    active_val = 0
    for t in active_trades:
        # Mark-to-Market: use current price if available, else buy price
        curr = state.get(t['symbol'], {}).get('current_price', t['buy_price'])
        active_val += t['quantity'] * curr
    return sim_balance + active_val

def export_snapshot():
    """Exports data for the Streamlit Dashboard"""
    global last_history_update
    try:
        # 1. Prepare Live Market Data
        monitored_data = {}
        for symbol, data in state.items():
            price = data.get("current_price", 0)
            if price == 0 and data.get('df') is not None and not data['df'].empty:
                price = data.get('df').iloc[-1]['close']
            
            latest_data = {}
            if data.get('df') is not None and not data['df'].empty:
                try:
                    latest_data = data['df'].iloc[-1].replace({np.nan: None}).to_dict()
                    latest_data = {str(k): v for k, v in latest_data.items()}
                except: pass

            monitored_data[symbol] = {
                "score": data.get("score"),
                "current_price": price,
                "price_change": data.get("price_change_percent"),
                "latest_data": latest_data
            }

        # 2. Prepare Active Trades
        active_safe = []
        for t in active_trades:
            t_safe = t.copy()
            active_safe.append(t_safe)
        
        # 3. Calculate Minute Performance (Every 60 Seconds)
        current_equity = compute_final_balance()
        now = time.time()
        
        if now - last_history_update >= 60:
            # Target Rate: (1 + 0.01)^(1/1440) - 1  ~= 0.00069%
            target_min_rate = (1.01 ** (1/1440)) - 1
            target_pct = target_min_rate * 100
            
            # Actual Rate
            actual_pct = 0.0
            if len(performance_history) > 0:
                prev_equity = performance_history[-1]['equity']
                if prev_equity > 0:
                    actual_pct = ((current_equity - prev_equity) / prev_equity) * 100
            
            performance_history.append({
                "timestamp": now,
                "equity": current_equity,
                "actual_return_pct": actual_pct,
                "target_return_pct": target_pct
            })
            last_history_update = now

        # 4. Save to JSON
        snapshot = {
            "timestamp": now,
            "balance": sim_balance,
            "equity": current_equity,
            "active_trades": active_safe,
            "monitored_tokens": monitored_data,
            "status_log": list(status_log),
            "performance_history": list(performance_history),
            "uptime": now - start_time
        }
        
        # Retry mechanism for file writing to avoid Access Denied errors
        for _ in range(5):
            try:
                with open("dashboard_data.json.tmp", "w") as f:
                    json.dump(snapshot, f, default=str)
                os.replace("dashboard_data.json.tmp", "dashboard_data.json")
                break
            except Exception:
                time.sleep(0.2)
                
    except Exception as e:
        logger.error(f"Export Error: {e}")

# --- Settings ---
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {"strategy": "Momentum-based"}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except: pass
    return DEFAULT_SETTINGS.copy()

# --- Indicators ---
def calculate_indicators(df):
    if df.empty or len(df) < 50: return df
    try:
        # Standardize columns
        for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c])
        
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean() - df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()

        # Bollinger Bands
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (2 * df['std_20'])
        df['BB_Lower'] = df['SMA_20'] - (2 * df['std_20'])

        # Volume SMA
        df['Vol_SMA_20'] = df['volume'].rolling(window=20).mean()

        # RSI
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/VOLATILITY_LOOKBACK, adjust=False).mean()
        
        # Support / Resistance (Donchian for Breakout)
        df['High_20'] = df['high'].rolling(20).max()
        df['Low_20'] = df['low'].rolling(20).min()
        
        if ENABLE_ICT:
            df['bullish_ob_low_recent'] = df['low'].shift(3) 
            df['Above_OB'] = df['close'] > df['bullish_ob_low_recent']

    except Exception as e:
        logger.error(f"Indicator Error: {e}")
    return df

# --- Strategies ---
def strategy_momentum(latest):
    score = 0
    criteria = []
    if latest['close'] > latest.get('EMA_200', 9e9): score += 2.0; criteria.append("Price>EMA200")
    if latest.get('EMA_20', 0) > latest.get('EMA_50', 1): score += 2.0; criteria.append("EMA_Golden")
    if 50 < latest.get('RSI_14', 50) < 70: score += 1.5; criteria.append("RSI_Bullish")
    if latest.get('MACD', 0) > latest.get('MACD_Signal', 0): score += 1.5; criteria.append("MACD_Cross")
    return score, criteria

def strategy_trend_following(latest):
    score = 0
    criteria = []
    if latest['close'] > latest.get('EMA_50', 9e9) > latest.get('EMA_200', 0): score += 3.0; criteria.append("Strong_Trend")
    if latest.get('EMA_20', 0) > latest.get('EMA_50', 0): score += 2.0; criteria.append("EMA_Alignment")
    if latest.get('MACD', 0) > 0: score += 1.0; criteria.append("MACD_Positive")
    return score, criteria

def strategy_breakout(latest):
    score = 0
    criteria = []
    if latest['close'] >= latest.get('High_20', 9e9): score += 3.0; criteria.append("20d_High_Break")
    if latest.get('volume', 0) > latest.get('Vol_SMA_20', 9e9) * 1.5: score += 2.0; criteria.append("High_Volume")
    if latest.get('RSI_14', 50) > 60: score += 1.0; criteria.append("RSI_Strength")
    return score, criteria

def strategy_mean_reversion(latest):
    score = 0
    criteria = []
    if latest['close'] < latest.get('BB_Lower', 0): score += 3.0; criteria.append("Below_BB_Lower")
    if latest.get('RSI_14', 50) < 30: score += 2.0; criteria.append("RSI_Oversold")
    if latest['close'] < latest.get('EMA_200', 0): score += 1.0; criteria.append("Below_EMA200")
    return score, criteria

def strategy_reversal(latest):
    score = 0
    criteria = []
    if latest.get('RSI_14', 50) < 25: score += 3.0; criteria.append("Deep_Oversold")
    if latest['close'] < latest.get('BB_Lower', 0): score += 2.0; criteria.append("Below_BB")
    if latest['close'] > latest['open']: score += 1.0; criteria.append("Bullish_Candle")
    return score, criteria

def strategy_scalping(latest):
    score = 0
    criteria = []
    if latest.get('EMA_5', 0) > latest.get('EMA_10', 0): score += 2.5; criteria.append("Fast_EMA_Cross")
    if 55 < latest.get('RSI_14', 50) < 75: score += 2.0; criteria.append("RSI_Momentum")
    if latest.get('volume', 0) > latest.get('Vol_SMA_20', 0): score += 1.5; criteria.append("Vol_Support")
    return score, criteria

# --- CCXT Tasks ---
async def monitor_token_rest(exchange, symbol):
    global sim_balance 
    if symbol not in state:
        state[symbol] = {"current_price": 0.0, "price_change_percent": 0.0, "df": None, "score": 0}
        
    while True:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            price = ticker['last']
            change = ticker['percentage']
            
            state[symbol]["current_price"] = price
            state[symbol]["price_change_percent"] = change
            
            # Trade Management
            for t in list(active_trades):
                if t['symbol'] == symbol:
                    if price <= t['stop_loss']:
                        sim_balance += (t['quantity'] * price)
                        active_trades.remove(t)
                        update_status(f"STOP LOSS: {symbol} @ {price:.4f}")
                    elif price >= t['take_profit']:
                        sim_balance += (t['quantity'] * price)
                        active_trades.remove(t)
                        update_status(f"TAKE PROFIT: {symbol} @ {price:.4f}")
                        
        except Exception as e:
            logger.debug(f"Ticker Error {symbol}: {e}")
            await asyncio.sleep(5)
            
        await asyncio.sleep(2)

async def analyze_token(exchange, symbol):
    global sim_balance 
    try:
        if len(active_trades) >= MAX_ACTIVE_TRADES: return
        
        ohlcv = await exchange.fetch_ohlcv(symbol, KLINE_INTERVAL, limit=KLINE_HISTORY_CANDLES)
        if not ohlcv: return

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = calculate_indicators(df)
        state[symbol]['df'] = df
        
        latest = df.iloc[-1]
        
        # Load Strategy
        settings = load_settings()
        strategy_name = settings.get("strategy", "Momentum-based")
        
        score = 0
        criteria = []
        
        if strategy_name == "Momentum-based":
            score, criteria = strategy_momentum(latest)
        elif strategy_name == "Trend-following":
            score, criteria = strategy_trend_following(latest)
        elif strategy_name == "Breakout":
            score, criteria = strategy_breakout(latest)
        elif strategy_name == "Mean reversion / range trading":
            score, criteria = strategy_mean_reversion(latest)
        elif strategy_name == "Reversal and pattern-based":
            score, criteria = strategy_reversal(latest)
        elif strategy_name == "Scalping-specific":
            score, criteria = strategy_scalping(latest)
        else:
            score, criteria = strategy_momentum(latest)

        # ICT Confluence
        if ENABLE_ICT and latest.get('Above_OB', False): 
            score += 1.0
            criteria.append("Above_OB")
        
        state[symbol]['score'] = round(score, 2)
        
        if score >= BUY_SCORE_THRESHOLD:
            if any(t['symbol'] == symbol for t in active_trades): return

            price = latest['close']
            atr = latest.get('ATR', price*0.02)
            invest = sim_balance * INVEST_FRACTION
            
            if sim_balance >= invest:
                sl = price - (atr * ATR_MULTIPLIER_SL)
                tp = price + ((price - sl) * RISK_REWARD_RATIO)
                
                trade = {
                    "symbol": symbol, "buy_price": price, "quantity": invest/price,
                    "invested": invest, "stop_loss": sl, "take_profit": tp,
                    "entry_criteria": criteria, "entry_score": score,
                    "strategy": strategy_name
                }
                
                sim_balance -= invest
                active_trades.append(trade)
                update_status(f"BUY ({strategy_name}): {symbol} @ {price:.4f} (Score: {score})")
            
    except Exception as e:
        logger.error(f"Analyze Error {symbol}: {e}")

async def run_analysis_loop(exchange):
    while True:
        await asyncio.sleep(ANALYSIS_INTERVAL_SECONDS)
        tasks = [analyze_token(exchange, s) for s in list(monitor_tasks.keys())]
        if tasks: await asyncio.gather(*tasks)

async def update_universe(exchange):
    while True:
        try:
            tickers = await exchange.fetch_tickers()
            valid = []
            for s, data in tickers.items():
                if '/USDT' in s and data['quoteVolume'] is not None:
                    if float(data['quoteVolume']) > MIN_24H_VOLUME_USDT:
                        valid.append(data)
            
            valid.sort(key=lambda x: abs(float(x.get('percentage', 0) or 0)), reverse=True)
            top = [x['symbol'] for x in valid[:UNIVERSE_SIZE]]
            
            current = set(monitor_tasks.keys())
            new = set(top)
            
            for s in current - new:
                monitor_tasks[s].cancel()
                del monitor_tasks[s]
                if s in state: del state[s]
                
            for s in new - current:
                monitor_tasks[s] = asyncio.create_task(monitor_token_rest(exchange, s))
                
            update_status(f"Universe Updated: {len(top)} tokens")
            
        except Exception as e:
            logger.error(f"Universe Error: {e}")
        
        await asyncio.sleep(300)

async def display_loop():
    while True:
        export_snapshot()
        console.clear()
        table = Table(title=f"CCXT BOT ({EXCHANGE_ID}) | Eq: ${compute_final_balance():.2f}")
        table.add_column("Symbol"); table.add_column("Price"); table.add_column("Score")
        
        items = list(state.items())
        items.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        for s, d in items[:10]:
            table.add_row(s, f"{d['current_price']:.4f}", str(d['score']))
            
        console.print(table)
        console.print(f"Active Trades: {len(active_trades)}")
        await asyncio.sleep(1)

# --- Backtesting Engine ---
async def backtest_strategy(exchange, symbol, days, strategy_name):
    try:
        # 1. Fetch Historical Data
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        ohlcv = await exchange.fetch_ohlcv(symbol, KLINE_INTERVAL, since=since, limit=1000)
        
        if not ohlcv: return None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 2. Calculate Indicators
        df = calculate_indicators(df)
        
        # 3. Simulate Strategy
        balance = SIM_STARTING_BALANCE
        trades = []
        active_trade = None
        equity_curve = []
        
        # Strategy Function Map
        strategies = {
            "Momentum-based": strategy_momentum,
            "Trend-following": strategy_trend_following,
            "Breakout": strategy_breakout,
            "Mean reversion / range trading": strategy_mean_reversion,
            "Reversal and pattern-based": strategy_reversal,
            "Scalping-specific": strategy_scalping
        }
        strategy_func = strategies.get(strategy_name, strategy_momentum)

        for i in range(50, len(df)):
            row = df.iloc[i]
            price = row['close']
            
            # Update Equity
            current_equity = balance
            if active_trade:
                current_equity += active_trade['quantity'] * price
            equity_curve.append({"timestamp": row.name, "equity": current_equity})
            
            # Check Exit
            if active_trade:
                if price <= active_trade['stop_loss'] or price >= active_trade['take_profit']:
                    balance += active_trade['quantity'] * price
                    trades.append({
                        "entry_price": active_trade['buy_price'],
                        "exit_price": price,
                        "pnl": (price - active_trade['buy_price']) * active_trade['quantity'],
                        "reason": "SL" if price <= active_trade['stop_loss'] else "TP"
                    })
                    active_trade = None
                continue # Don't enter and exit in same candle
            
            # Check Entry
            score, criteria = strategy_func(row)
            
            # ICT Confluence
            if ENABLE_ICT and row.get('Above_OB', False): score += 1.0
            
            if score >= BUY_SCORE_THRESHOLD:
                atr = row.get('ATR', price*0.02)
                invest = balance * INVEST_FRACTION
                if balance >= invest:
                    sl = price - (atr * ATR_MULTIPLIER_SL)
                    tp = price + ((price - sl) * RISK_REWARD_RATIO)
                    
                    active_trade = {
                        "buy_price": price,
                        "quantity": invest / price,
                        "stop_loss": sl,
                        "take_profit": tp
                    }
                    balance -= invest
        
        # Close final position
        if active_trade:
            price = df.iloc[-1]['close']
            balance += active_trade['quantity'] * price
            
        return {
            "strategy": strategy_name,
            "final_balance": balance,
            "return_pct": ((balance - SIM_STARTING_BALANCE) / SIM_STARTING_BALANCE) * 100,
            "trades_count": len(trades),
            "equity_curve": equity_curve
        }

    except Exception as e:
        logger.error(f"Backtest Error {strategy_name}: {e}")
        return None

# --- Main ---
async def main():
    update_status(f"Starting CCXT Engine ({EXCHANGE_ID})...")
    
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    
    # Define connection parameters
    params = {
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    }
    
    # Inject Keys only if they exist
    if API_KEY is not None:
        params['apiKey'] = API_KEY
        params['secret'] = API_SECRET
        if API_PASSWORD: params['password'] = API_PASSWORD
    else:
        update_status("Public Mode: Connecting without API keys.")

    exchange = exchange_class(params)
    
    if SIMULATION_MODE:
        if hasattr(exchange, 'set_sandbox_mode'):
            exchange.set_sandbox_mode(True)
        update_status("Note: Running in Simulation Mode (Internal Balance)")

    try:
        await asyncio.gather(
            update_universe(exchange),
            run_analysis_loop(exchange),
            display_loop()
        )
    finally:
        await exchange.close()

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())