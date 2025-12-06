# config.py

# --- Exchange Setup ---
EXCHANGE_ID = 'binance'         # 'binance', 'kraken', 'kucoin', 'bybit', etc.

# SET THESE TO None TO USE PUBLIC DATA (NO ACCOUNT REQUIRED)
API_KEY = None
API_SECRET = None
API_PASSWORD = "" 

# --- Simulation Settings ---
SIMULATION_MODE = False         # False = Use Real Market Prices. True = Mock/Fake Data.
SIM_STARTING_BALANCE = 1000.0   # Your paper trading balance
INVEST_FRACTION = 0.10          # Invest 10% per trade
MAX_ACTIVE_TRADES = 5           
FEE_RATE = 0.001                

# --- Screener & Universe ---
UNIVERSE_SIZE = 15              
TOKEN_WHITELIST = []            # e.g. ['BTC/USDT', 'ETH/USDT']
MIN_24H_VOLUME_USDT = 1000000   
MIN_24H_CHANGE_PCT = 1.5        

# --- Strategy: Risk Management ---
RISK_REWARD_RATIO = 1.5         
STOP_LOSS_PCT = 0.02            
BUY_SCORE_THRESHOLD = 6.0       
DYNAMIC_POSITION_SIZING = True  
VOLATILITY_LOOKBACK = 14        
ATR_MULTIPLIER_SL = 1.5         

# --- Strategy: Timeframes ---
KLINE_INTERVAL = "15m"          
KLINE_HISTORY_CANDLES = 200     # Number of candles to fetch
ANALYSIS_INTERVAL_SECONDS = 60  
MAX_TRADE_DURATION_CANDLES = 50 

# --- Indicators ---
ENABLE_MACD = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ENABLE_MAE = True               
MAE_LENGTH = 20
MAE_PCT_OFFSET = 0.02           

ENABLE_ICT = True               

# --- System ---
PERFORMANCE_LOG_FILE = "strategy_performance.json"
SELF_LEARNING_ENABLED = True