import streamlit as st
import pandas as pd
import json
import time
import os
import asyncio
import ccxt.async_support as ccxt
from trading_bot import backtest_strategy, EXCHANGE_ID

st.set_page_config(page_title="CCXT Bot Dashboard", page_icon="🌐", layout="wide")

def load_data():
    try:
        with open("dashboard_data.json", "r") as f:
            return json.load(f)
    except: return None

# --- Settings Management ---
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {"strategy": "Momentum-based"}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except: pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

settings = load_settings()

st.title("🌐 CCXT Algo-Trading Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Bot Settings")
    
    strategy_options = [
        "Momentum-based",
        "Trend-following",
        "Breakout",
        "Mean reversion / range trading",
        "Reversal and pattern-based",
        "Scalping-specific"
    ]
    
    current_strategy = settings.get("strategy", "Momentum-based")
    if current_strategy not in strategy_options:
        current_strategy = strategy_options[0]
        
    selected_strategy = st.selectbox(
        "Trading Strategy", 
        strategy_options, 
        index=strategy_options.index(current_strategy)
    )
    
    if selected_strategy != settings.get("strategy"):
        settings["strategy"] = selected_strategy
        save_settings(settings)
        st.success(f"Strategy updated to: {selected_strategy}")
        time.sleep(1)
        st.rerun()
        
    st.divider()
    st.info(f"Current Strategy:\n**{settings.get('strategy')}**")

data = load_data()

# --- Top Metrics ---
if data:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity", f"${data.get('equity', 0):,.2f}")
    m2.metric("Balance", f"${data.get('balance', 0):,.2f}")
    m3.metric("Trades", len(data.get('active_trades', [])))
    uptime_min = int(data.get('uptime', 0)//60)
    m4.metric("Uptime", f"{uptime_min}m")

    # --- Performance Graph (Minute Rate) ---
    st.subheader("⏱️ Minute-by-Minute Return (%)")
    perf_hist = data.get('performance_history', [])

    if perf_hist:
        df_hist = pd.DataFrame(perf_hist)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='s')
        df_hist.set_index('timestamp', inplace=True)
        
        # Rename columns for the graph
        df_hist = df_hist.rename(columns={
            'actual_return_pct': 'Actual Return %',
            'target_return_pct': 'Target (0.00069%)'
        })
        
        # Display the graph comparing Actual vs Target
        st.line_chart(df_hist[['Actual Return %', 'Target (0.00069%)']], height=300)
    else:
        st.info("Waiting for the first minute of data to calculate rates...")
else:
    st.warning("Waiting for bot data... Ensure 'trading_bot.py' is running.")

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Active Trades", "🔍 Market Scanner", "📜 Logs", "🧪 Backtesting"])

if data:
    with tab1:
        trades = data.get('active_trades', [])
        if trades:
            df = pd.DataFrame(trades)
            cols_to_show = ['symbol', 'buy_price', 'take_profit', 'stop_loss', 'invested', 'entry_score', 'strategy']
            available_cols = [c for c in cols_to_show if c in df.columns]
            df = df[available_cols]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No active trades.")

    with tab2:
        monitored = data.get('monitored_tokens', {})
        rows = []
        for s, d in monitored.items():
            rows.append({
                "Symbol": s, 
                "Price": d['current_price'], 
                "Score": d['score'], 
                "Change %": d['price_change']
            })
        
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

    with tab3:
        for log in reversed(data.get('status_log', [])):
            st.text(log)

# --- Backtesting Tab ---
with tab4:
    st.header("🧪 Strategy Backtester")
    st.markdown("Run simulations on historical data to compare strategy performance.")
    
    c1, c2, c3 = st.columns(3)
    bt_symbol = c1.text_input("Symbol", "BTC/USDT")
    bt_days = c2.number_input("Days to Backtest", min_value=1, max_value=30, value=7)
    
    if st.button("Run Backtest (All Strategies)"):
        st.info(f"Fetching data for {bt_symbol} (Last {bt_days} days)...")
        
        async def run_tests():
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            exchange = exchange_class({'enableRateLimit': True})
            results = []
            try:
                tasks = []
                for strat in strategy_options:
                    tasks.append(backtest_strategy(exchange, bt_symbol, bt_days, strat))
                
                results = await asyncio.gather(*tasks)
            finally:
                await exchange.close()
            return results

        results = asyncio.run(run_tests())
        
        # Display Results
        summary = []
        for res in results:
            if res:
                summary.append({
                    "Strategy": res['strategy'],
                    "Return %": f"{res['return_pct']:.2f}%",
                    "Final Balance": f"${res['final_balance']:.2f}",
                    "Trades": res['trades_count']
                })
                
                # Plot Equity Curve
                if res['equity_curve']:
                    df_eq = pd.DataFrame(res['equity_curve'])
                    df_eq.set_index('timestamp', inplace=True)
                    st.subheader(f"{res['strategy']} Equity Curve")
                    st.line_chart(df_eq['equity'])

        if summary:
            st.subheader("🏆 Performance Summary")
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        else:
            st.error("No data returned. Check symbol or connection.")

if not data:
    time.sleep(2)
    st.rerun()
else:
    time.sleep(5)
    st.rerun()