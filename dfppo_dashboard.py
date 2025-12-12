import json
import os
import time

import pandas as pd
import streamlit as st


METRICS_FILE = "dfppo_metrics.json"


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return []
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


st.set_page_config(page_title="DF-PPO Training Dashboard", layout="wide")
st.title("DF-PPO Training Dashboard")

metrics = load_metrics()

if not metrics:
    st.warning("No DF-PPO metrics found yet. Make sure train_dfppo_local.py is running.")
else:
    df = pd.DataFrame(metrics)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)

    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", int(latest.get("version", 0)))
    c2.metric("Train Sharpe", f"{latest.get('train_sharpe', 0.0):.2f}")
    c3.metric("Test Sharpe", f"{latest.get('test_sharpe', 0.0):.2f}")
    c4.metric("Test MDD", f"{latest.get('test_mdd', 0.0):.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Compute Score", int(latest.get("compute_score", 0)))
    c6.metric("Role", str(latest.get("role", "unknown")))
    if "entropy" in latest:
        c7.metric("Policy Entropy", f"{latest.get('entropy', 0.0):.3f}")
    else:
        c7.metric("Policy Entropy", "n/a")

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
