import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import fetch_binance_klines, get_close_prices, validate_hourly_data
from src.model import predict_next_range

from supabase import create_client
from datetime import datetime, timezone

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="BTC Next-Hour Forecast",
    page_icon="₿",
    layout="wide",
)


BACKTEST_PATH = Path("outputs/backtest_results.jsonl")


@st.cache_data(ttl=300)
def load_live_data(limit: int = 600):
    df = fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=limit)
    prices = get_close_prices(df)
    validate_hourly_data(prices)
    return df, prices


@st.cache_data(ttl=300)
def load_backtest_metrics():
    if not BACKTEST_PATH.exists():
        return None

    rows = []
    with BACKTEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    results = pd.DataFrame(rows)

    return {
        "coverage_95": results["covered_95"].mean(),
        "average_width_95": results["width_95"].mean(),
        "mean_winkler_95": results["winkler_95"].mean(),
        "n_predictions": len(results),
    }


def make_chart(prices: pd.Series, low: float, high: float):
    recent = prices.tail(50)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            mode="lines",
            name="BTC close",
        )
    )

    next_time = recent.index[-1] + pd.Timedelta(hours=1)

    fig.add_trace(
        go.Scatter(
            x=[next_time, next_time],
            y=[low, high],
            mode="lines",
            name="Predicted 95% range",
            line=dict(width=8),
        )
    )

    fig.add_hrect(
        y0=low,
        y1=high,
        fillcolor="rgba(0, 150, 255, 0.15)",
        line_width=0,
        annotation_text="Next-hour 95% range",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Last 50 BTCUSDT hourly closes + next-hour forecast range",
        xaxis_title="Time",
        yaxis_title="BTCUSDT price",
        height=520,
        hovermode="x unified",
    )

    return fig


st.title("₿ BTCUSDT Next-Hour 95% Forecast")
st.caption("Live hourly BTC forecast using Binance public data + Student-t GBM simulation")

df, prices = load_live_data(limit=600)

current_price = float(prices.iloc[-1])
latest_bar_time = prices.index[-1]

low_95, high_95 = predict_next_range(
    prices.tail(500),
    n_sims=5000,
    vol_window=80,
    drift_window=200,
    tail_df=4,
    range_scale=1.05,
    seed=42,
)


prediction_time = datetime.now(timezone.utc).isoformat()
target_time = (prices.index[-1] + pd.Timedelta(hours=1)).floor("h").isoformat()

if "last_saved_target" not in st.session_state or st.session_state["last_saved_target"] != target_time:

    supabase.table("prediction_history").insert({
        "prediction_time": prediction_time,
        "target_time": target_time,
        "current_price": current_price,
        "low_95": low_95,
        "high_95": high_95,
        "range_width": high_95 - low_95,
    }).execute()

    st.session_state["last_saved_target"] = target_time

# Update actual prices for past predictions
full_df = fetch_binance_klines(limit=1500)
full_prices = get_close_prices(full_df)

history = supabase.table("prediction_history").select("*").execute()

for row in history.data:
    if row["actual_price"] is None:
        try:
            actual_price = full_prices.loc[pd.to_datetime(row["target_time"])]
            covered = row["low_95"] <= actual_price <= row["high_95"]

            supabase.table("prediction_history") \
                .update({
                    "actual_price": float(actual_price),
                    "covered_95": covered
                }) \
                .eq("id", row["id"]) \
                .execute()
        except Exception as e:
            print("Update error:", e)


metrics = load_backtest_metrics()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current BTC price", f"${current_price:,.2f}")
col2.metric("Predicted low 95%", f"${low_95:,.2f}")
col3.metric("Predicted high 95%", f"${high_95:,.2f}")
col4.metric("Range width", f"${high_95 - low_95:,.2f}")

st.write(f"Latest closed hourly bar: `{latest_bar_time}`")

if metrics:
    st.subheader("Backtest metrics — last 720 hourly predictions")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage 95%", f"{metrics['coverage_95']:.4f}")
    m2.metric("Average width 95%", f"${metrics['average_width_95']:,.2f}")
    m3.metric("Mean Winkler 95", f"{metrics['mean_winkler_95']:,.2f}")
    m4.metric("Predictions", f"{metrics['n_predictions']}")
else:
    st.warning("Backtest file not found. Run `python src/backtest.py` first.")

st.plotly_chart(make_chart(prices, low_95, high_95), use_container_width=True)

with st.expander("Model details"):
    st.write(
        """
        - Data source: Binance public klines endpoint
        - Symbol: BTCUSDT
        - Interval: 1 hour
        - Model: one-step GBM simulation with Student-t fat-tailed shocks
        - Volatility window: 80 hourly returns
        - Drift window: 200 hourly returns
        - Range scale: 1.05
        - Backtest: no-peeking rolling evaluation
        """
    )

st.subheader("Prediction History")

history = supabase.table("prediction_history") \
    .select("*") \
    .order("prediction_time", desc=True) \
    .limit(50) \
    .execute()

if history.data:
    df_hist = pd.DataFrame(history.data)
    st.dataframe(df_hist.sort_values("prediction_time", ascending=False))
else:
    st.info("No predictions yet")