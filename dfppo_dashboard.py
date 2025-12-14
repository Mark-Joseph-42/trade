import os
import time
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Configuration
LOGS_DIR = Path("logs")
REFRESH_INTERVAL = 5  # seconds

def get_latest_log_file():
    """Find the most recent log file in the logs directory."""
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return None
    
    log_files = list(LOGS_DIR.glob("training_*.csv"))
    if not log_files:
        return None
    
    # Get the most recently modified file
    return max(log_files, key=os.path.getmtime)

def load_metrics():
    """Load metrics from the latest log file."""
    log_file = get_latest_log_file()
    
    if not log_file or not log_file.exists():
        return pd.DataFrame()
    
    try:
        # Read the CSV file, handling potential empty files
        df = pd.read_csv(log_file)
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.warning(f"Error reading log file: {e}")
        return pd.DataFrame()

# Set up the page
st.set_page_config(
    page_title="DF-PPO Training Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DF-PPO Training Dashboard")

# Auto-refresh control in sidebar
st.sidebar.title("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 60, 5)

# Load metrics
df = load_metrics()

if df.empty:
    st.warning("No training data found. Make sure the training script is running.")
    st.stop()

# Display latest metrics
st.sidebar.subheader("Latest Metrics")
latest = df.iloc[-1]

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

# Display key metrics
with col1:
    st.metric("Episode", int(latest.get('episode', 0)) if 'episode' in df.columns else 'N/A')
    st.metric("Step", int(latest.get('step', 0)) if 'step' in df.columns else 'N/A')

with col2:
    if 'equity' in df.columns:
        st.metric("Equity", f"${latest.get('equity', 0):.2f}")
    if 'position' in df.columns:
        st.metric("Position", latest.get('position', 'N/A'))

with col3:
    if 'reward' in df.columns:
        st.metric("Reward", f"{latest.get('reward', 0):.4f}")
    if 'drawdown' in df.columns:
        st.metric("Drawdown", f"{latest.get('drawdown', 0)*100:.2f}%")

with col4:
    if 'loss' in df.columns:
        st.metric("Loss", f"{latest.get('loss', 0):.6f}" if pd.notna(latest.get('loss')) else 'N/A')

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Equity Curve", "Rewards", "Drawdown"])

with tab1:
    if 'equity' in df.columns:
        st.line_chart(df.set_index('timestamp')['equity'])
    else:
        st.warning("No equity data available")

with tab2:
    if 'reward' in df.columns:
        st.line_chart(df.set_index('timestamp')['reward'])
    else:
        st.warning("No reward data available")

with tab3:
    if 'drawdown' in df.columns:
        st.line_chart(df.set_index('timestamp')['drawdown'])
    else:
        st.warning("No drawdown data available")

# Display raw data
expander = st.expander("View Raw Data")
expander.dataframe(df, use_container_width=True)

# Auto-refresh
st_autorefresh(interval=refresh_rate * 1000, key="data_refresh")

if {"timestamp", "train_sharpe", "test_sharpe"}.issubset(df.columns):
    st.subheader("Sharpe Ratio Over Time")
    sharpe_df = df[["timestamp", "train_sharpe", "test_sharpe"]].set_index("timestamp")
    sharpe_df = sharpe_df.rename(
        columns={"train_sharpe": "Train Sharpe", "test_sharpe": "Test Sharpe"}
    )
    st.line_chart(sharpe_df)

if {"timestamp", "train_mdd", "test_mdd"}.issubset(df.columns):
    st.subheader("Max Drawdown Over Time")
    mdd_df = df[["timestamp", "train_mdd", "test_mdd"]].set_index("timestamp")
    mdd_df = mdd_df.rename(
        columns={"train_mdd": "Train MDD", "test_mdd": "Test MDD"}
    )
    st.line_chart(mdd_df)

if {"timestamp", "entropy"}.issubset(df.columns):
    st.subheader("Policy Entropy Over Time")
    ent_df = df[["timestamp", "entropy"]].set_index("timestamp")
    ent_df = ent_df.rename(columns={"entropy": "Policy Entropy"})
    st.line_chart(ent_df)

    st.subheader("Raw Metrics")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    time.sleep(5)
    st.rerun()
