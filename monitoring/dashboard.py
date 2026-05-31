"""
monitoring/dashboard.py
Dashboard monitoring Streamlit untuk sistem ML Forex Advisor.

Jalankan dengan:
    streamlit run monitoring/dashboard.py
"""
import sys
import time
import json
import yaml
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

# Add project root to sys.path so monitoring/dashboard.py can import local packages
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.collector import MT5Collector

# =====================================================================
# Konfigurasi halaman
# =====================================================================
st.set_page_config(
    page_title="ML Forex Advisor — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load config
@st.cache_resource
def load_config():
    try:
        with open("config/config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "api": {"host": "127.0.0.1", "port": 5000},
            "data": {"primary_symbol": "EURUSD"},
            "monitoring": {"min_win_rate": 0.45, "min_sharpe": 0.5},
        }

cfg = load_config()
API_BASE = f"http://{cfg['api']['host']}:{cfg['api']['port']}"

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.title("⚙️ Kontrol")
    selected_symbol = st.selectbox(
        "Symbol",
        cfg["data"].get("symbols", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]),
        index=0,
    )
    timeframe = st.selectbox("Timeframe", ["M5", "M15", "H1", "H4", "D1"], index=0)
    auto_refresh = st.toggle("Auto Refresh (30s)", value=False)
    refresh_btn  = st.button("🔄 Refresh Sekarang", use_container_width=True)

    st.divider()
    st.subheader("Pengaturan Tampilan")
    n_trades_chart = st.slider("Jumlah trade di chart", 20, 200, 50)
    equity_bars    = st.slider("Bar equity curve", 100, 1000, 300)

    st.divider()
    st.subheader("Model Actions")
    if st.button("🚀 Trigger Retrain", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/retrain?symbol={selected_symbol}", timeout=5)
            st.success("Retrain dimulai!") if r.status_code == 200 else st.error(f"Error: {r.text}")
        except Exception as e:
            st.error(f"Tidak dapat terhubung ke API: {e}")

    st.divider()
    st.subheader("🧪 Tick Backtest")
    backtest_mode = st.radio("Mode Backtest", ["Live", "By Date Range"])
    
    if backtest_mode == "By Date Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.strptime(cfg["data"].get("tick_backtest_start_date", "2025-01-01"), "%Y-%m-%d"),
                label_visibility="collapsed"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.strptime(cfg["data"].get("tick_backtest_end_date", "2026-12-31"), "%Y-%m-%d"),
                label_visibility="collapsed"
            )
        
        st.write(f"📅 Range: {start_date} → {end_date}")
        
        if st.button("▶️ Jalankan Tick Backtest", use_container_width=True):
            st.info(f"Tick backtest untuk {selected_symbol} | {start_date} to {end_date}")
            st.info(f"Leverage: 1:{cfg['backtest'].get('leverage', 1)}")
            st.success("Backtest siap dijalankan di CLI:\n" + 
                      f"`python main.py --mode ticktest --symbol {selected_symbol} --start-date {start_date} --end-date {end_date} --tick-timeframe {timeframe}`")
    
    st.divider()
    st.caption(f"API: `{API_BASE}`")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"⚙️ Leverage: 1:{cfg['backtest'].get('leverage', 1)} | Initial: ${cfg['backtest'].get('initial_balance', 10000)}")

# =====================================================================
# Data generators (simulasi jika API tidak tersedia)
# =====================================================================

def get_api_status():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return {"status": "offline", "models_loaded": False}, False


def get_model_info():
    try:
        r = requests.get(f"{API_BASE}/model/info", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_live_signal(symbol, timeframe):
    try:
        r = requests.get(f"{API_BASE}/predict/live?symbol={symbol}&timeframe={timeframe}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def aggregate_tick_ohlcv(df_ticks: pd.DataFrame, timeframe: str, n: int) -> pd.DataFrame:
    tf_map = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
    }
    tf = tf_map.get(timeframe.upper(), timeframe.lower())
    df = df_ticks.copy()
    df = df.sort_index()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    bars = df["mid"].resample(tf).ohlc()
    bars["volume"] = df["bid"].resample(tf).count().astype(int)
    bars = bars.dropna()
    return bars.tail(n)


def generate_demo_data(cfg, symbol, timeframe, n=300):
    """Data demo jika API tidak tersedia. Coba pakai tick history nyata dulu."""
    df_ohlc = None
    try:
        collector = MT5Collector(cfg)
        df_ticks = collector.load_tick_history(symbol)
        # Pakai hanya tick 2025-2026 untuk dashboard real tick history
        df_ticks.index = pd.to_datetime(df_ticks.index, utc=True)
        # Sort index first to avoid "non-monotonic DatetimeIndex" error, then filter by date range
        df_ticks = df_ticks.sort_index()
        start_filter = pd.to_datetime("2025-01-01", utc=True)
        end_filter = pd.to_datetime("2026-12-31", utc=True)
        df_ticks = df_ticks[(df_ticks.index >= start_filter) & (df_ticks.index <= end_filter)]
        if df_ticks.empty:
            raise ValueError("Tidak ada tick 2025-2026 di history.")

        df_ohlc = aggregate_tick_ohlcv(df_ticks, timeframe, n)
        df_ohlc = df_ohlc.reset_index()
        if "Unnamed: 0" in df_ohlc.columns and pd.api.types.is_datetime64_any_dtype(df_ohlc["Unnamed: 0"]):
            df_ohlc["time"] = df_ohlc["Unnamed: 0"]
            df_ohlc = df_ohlc.drop(columns=["Unnamed: 0"])
        if "time" not in df_ohlc.columns:
            if "index" in df_ohlc.columns:
                df_ohlc.rename(columns={"index": "time"}, inplace=True)
            else:
                df_ohlc["time"] = df_ohlc.index
        df_ohlc["time"] = pd.to_datetime(df_ohlc["time"], utc=True, errors="coerce")
        if df_ohlc["time"].isna().any():
            raise ValueError("Time column pada candle tidak valid setelah konversi.")
        if df_ohlc.empty:
            raise ValueError("Tick history tersedia tapi tidak menghasilkan candle yang cukup.")
    except Exception as exc:
        st.warning(f"Tidak dapat memuat tick history nyata: {exc}. Menggunakan data dummy.")

    if df_ohlc is None or df_ohlc.empty:
        np.random.seed(int(time.time()) % 100)
        dates = pd.date_range(end=datetime.now(), periods=n, freq="1h")
        close_prices = 1.08 + np.cumsum(np.random.randn(n) * 0.0005)
        ohlc_data = []
        for i in range(n):
            close = close_prices[i]
            open_p = close + np.random.randn() * 0.0002
            high = max(open_p, close) + abs(np.random.randn() * 0.0003)
            low = min(open_p, close) - abs(np.random.randn() * 0.0003)
            ohlc_data.append({
                "time": dates[i],
                "open": round(open_p, 5),
                "high": round(high, 5),
                "low": round(low, 5),
                "close": round(close, 5),
                "volume": int(np.random.uniform(1000, 5000)),
            })
        df_ohlc = pd.DataFrame(ohlc_data)
        close_prices = df_ohlc["close"].tolist()
        dates = df_ohlc["time"].tolist()
        demo_mode = True
    else:
        demo_mode = False
        close_prices = df_ohlc["close"].tolist()
        dates = pd.to_datetime(df_ohlc["time"]).tolist()

    balance = 10000.0
    equity_curve = [balance]
    trades = []
    signals = []
    for i in range(len(df_ohlc)):
        if i == 0:
            signals.append("HOLD")
            continue
        prev = df_ohlc["close"].iloc[i - 1]
        curr = df_ohlc["close"].iloc[i]
        if curr > prev:
            signals.append("BUY")
        elif curr < prev:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    for i in range(1, len(df_ohlc)):
        if signals[i] != "HOLD" and len(trades) < 100:
            entry_price = float(df_ohlc["close"].iloc[i])
            direction = signals[i]
            win = np.random.random() < 0.52
            pnl = (0.0002 if win else -0.00012) * 10000
            balance += pnl
            if direction == "BUY":
                sl_price = round(entry_price - 0.0025, 5)
                tp_price = round(entry_price + 0.0038, 5)
            else:
                sl_price = round(entry_price + 0.0025, 5)
                tp_price = round(entry_price - 0.0038, 5)
            trades.append({
                "time": df_ohlc["time"].iloc[i],
                "symbol": symbol,
                "direction": direction,
                "pnl": round(pnl, 2),
                "result": "WIN" if win else "LOSS",
                "confidence": round(np.random.uniform(0.55, 0.95), 3),
                "balance_after": round(balance, 2),
                "entry_price": round(entry_price, 5),
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_pips": 25,
                "tp_pips": 38,
                "lot_size": 0.05,
            })
        equity_curve.append(balance + np.random.randn() * 30)

    df_equity = pd.DataFrame({"time": dates[: len(equity_curve)], "equity": equity_curve})
    df_trades = pd.DataFrame(trades)
    model_metrics = {
        "val_accuracy": 0.58,
        "sharpe_ratio": 1.24,
        "max_drawdown": -0.082,
        "win_rate": (df_trades["result"] == "WIN").sum() / len(df_trades) if len(df_trades) > 0 else 0,
        "profit_factor": 1.41,
        "total_trades": len(df_trades),
        "total_pnl": round(balance - 10000, 2),
    }
    last_close = close_prices[-1] if len(close_prices) > 0 else 0.0
    last_signal = signals[-1] if len(signals) > 0 else "HOLD"
    signal_now = {
        "signal": last_signal,
        "confidence": round(np.random.uniform(0.55, 0.90), 3),
        "sl_price": round(last_close - 0.0025, 5),
        "tp_price": round(last_close + 0.0038, 5),
        "sl_pips": 25,
        "tp_pips": 38,
        "lot_size": 0.05,
        "current_price": round(last_close, 5),
        "risk_passed": True,
        "proba_hold": 0.22,
        "proba_buy": 0.43,
        "proba_sell": 0.35,
        "atr_value": 0.00095,
    }
    return df_equity, df_trades, model_metrics, signal_now, close_prices, dates, df_ohlc

# =====================================================================
# Fetch data
# =====================================================================
api_status, api_online = get_api_status()

if api_online:
    signal_data = get_live_signal(selected_symbol, timeframe)
    model_info  = get_model_info()
    demo_mode = False
    st.toast("✅ Terhubung ke API server", icon="🟢")
else:
    demo_mode = True
    # Ensure model_info is defined even when API is offline
    model_info = None

df_equity, df_trades, model_metrics, signal_now, close_prices, dates, df_ohlc = generate_demo_data(cfg, selected_symbol, timeframe, equity_bars)

if demo_mode:
    st.warning("⚠️  API server offline — menampilkan data demo. Jalankan `python main.py --mode api` untuk data live.", icon="⚠️")
    signal_data = signal_now

# =====================================================================
# Header
# =====================================================================
st.title("📊 ML Forex Advisor — Monitoring Dashboard")
col_sym, col_tf, col_status = st.columns([2, 1, 2])
with col_sym:
    st.subheader(f"Symbol: **{selected_symbol}**")
with col_tf:
    st.caption(f"Timeframe: {timeframe}")
with col_status:
    if api_online:
        st.success("🟢 API Online | Model: " + ("✅ Loaded" if api_status.get("models_loaded") else "⏳ Loading"))
    else:
        st.error("🔴 API Offline (Demo Mode)")

st.divider()

# =====================================================================
# Sinyal Terkini — Signal Panel
# =====================================================================
sig = signal_data or signal_now
signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig.get("signal", "HOLD"), "⚪")
signal_label = sig.get("signal", "HOLD")
conf_pct     = int(sig.get("confidence", 0) * 100)

st.subheader("📡 Sinyal Terkini")
sig_c1, sig_c2, sig_c3, sig_c4, sig_c5, sig_c6 = st.columns(6)

with sig_c1:
    color_bg = {"BUY": "#1a7a3c", "SELL": "#7a1a1a", "HOLD": "#4a4a4a"}.get(signal_label, "#4a4a4a")
    st.metric("Signal", f"{signal_color} {signal_label}")

with sig_c2:
    st.metric("Confidence", f"{conf_pct}%",
              delta="✅ Pass" if sig.get("risk_passed") else "❌ Fail")

with sig_c3:
    st.metric("Harga Saat Ini", str(sig.get("current_price", "—")))

with sig_c4:
    st.metric("Stop Loss", f"{sig.get('sl_price', '—')} ({sig.get('sl_pips', 0)} pips)")

with sig_c5:
    st.metric("Take Profit", f"{sig.get('tp_price', '—')} ({sig.get('tp_pips', 0)} pips)")

with sig_c6:
    st.metric("Lot Size", str(sig.get("lot_size", "—")))

# Probability bar
st.markdown("**Probabilitas Kelas**")
prob_c1, prob_c2, prob_c3 = st.columns(3)
with prob_c1:
    st.progress(sig.get("proba_buy", 0), text=f"BUY: {sig.get('proba_buy', 0)*100:.1f}%")
with prob_c2:
    st.progress(sig.get("proba_sell", 0), text=f"SELL: {sig.get('proba_sell', 0)*100:.1f}%")
with prob_c3:
    st.progress(sig.get("proba_hold", 0), text=f"HOLD: {sig.get('proba_hold', 0)*100:.1f}%")

st.divider()

# =====================================================================
# Konfigurasi Backtest dengan Leverage
# =====================================================================
st.subheader("⚙️ Backtest Configuration")
bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
with bt_c1:
    st.metric("Initial Balance", f"${cfg['backtest'].get('initial_balance', 10000):,.2f}")
with bt_c2:
    leverage = cfg['backtest'].get('leverage', 1)
    st.metric("Leverage", f"1:{leverage}")
with bt_c3:
    margin_req = 100 / leverage
    st.metric("Margin Requirement", f"{margin_req:.3f}%")
with bt_c4:
    st.metric("Pip Value", f"${cfg['backtest'].get('pip_value', 10.0)}")

st.info(
    f"**Tick Backtest Range:** {cfg['data'].get('tick_backtest_start_date', '2025-01-01')} to {cfg['data'].get('tick_backtest_end_date', '2026-12-31')}\n\n"
    f"Run backtest dengan leverage 1:{leverage} menggunakan: "
    f"`python main.py --mode ticktest --symbol {selected_symbol} --start-date 2025-01-01 --end-date 2026-12-31`"
)

st.divider()

# =====================================================================
# Metrik Performa
# =====================================================================
st.subheader("📈 Metrik Performa")
m = model_metrics
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

def delta_color(val, threshold, invert=False):
    good = val >= threshold if not invert else val <= threshold
    return f"+{val}" if good else f"{val}"

with m1:
    wr = m.get("win_rate", 0)
    st.metric("Win Rate", f"{wr*100:.1f}%",
              delta="✅" if wr >= cfg["monitoring"].get("min_win_rate", 0.45) else "⚠️")
with m2:
    sr = m.get("sharpe_ratio", 0)
    st.metric("Sharpe Ratio", f"{sr:.2f}",
              delta="✅" if sr >= cfg["monitoring"].get("min_sharpe", 0.5) else "⚠️")
with m3:
    dd = m.get("max_drawdown", 0)
    st.metric("Max Drawdown", f"{dd*100:.1f}%",
              delta="✅" if dd > -0.10 else "⚠️ Tinggi")
with m4:
    pf = m.get("profit_factor", 0)
    st.metric("Profit Factor", f"{pf:.2f}",
              delta="✅" if pf >= 1.2 else "⚠️")
with m5:
    st.metric("Total Trades", m.get("total_trades", 0))
with m6:
    pnl = m.get("total_pnl", 0)
    st.metric("Total PnL", f"${pnl:,.2f}",
              delta=f"+{pnl:.0f}" if pnl > 0 else f"{pnl:.0f}")
with m7:
    acc = m.get("val_accuracy", 0)
    st.metric("Model Accuracy", f"{acc*100:.1f}%")

st.divider()

# =====================================================================
# Charts — Equity Curve + Price + Trade Distribution + Candlestick
# =====================================================================
chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
    "📈 Equity Curve", "🕯️ Candlestick & Trade", "🎯 Distribusi Trade", "🔬 Model Analysis", "📊 Trade History"
])

# ---- Tab 1: Equity Curve ----
with chart_tab1:
    fig_eq = go.Figure()

    # Equity line
    fig_eq.add_trace(go.Scatter(
        x=df_equity["time"],
        y=df_equity["equity"].round(2),
        mode="lines",
        name="Equity",
        line=dict(color="#3266ad", width=2),
        fill="tonexty",
        fillcolor="rgba(50, 102, 173, 0.08)",
    ))

    # Balance baseline
    fig_eq.add_hline(y=10000, line_dash="dot", line_color="gray", annotation_text="Initial Balance $10,000")

    # Underwater / drawdown shading
    peak = df_equity["equity"].cummax()
    dd_series = (df_equity["equity"] - peak).round(2)
    fig_eq.add_trace(go.Scatter(
        x=df_equity["time"],
        y=dd_series,
        mode="lines",
        name="Drawdown",
        line=dict(color="#e24b4a", width=1),
        yaxis="y2",
        fill="tozeroy",
        fillcolor="rgba(226, 75, 74, 0.08)",
    ))

    fig_eq.update_layout(
        title=f"Equity Curve — {selected_symbol}",
        yaxis=dict(title="Equity ($)", side="left"),
        yaxis2=dict(title="Drawdown ($)", side="right", overlaying="y", showgrid=False),
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

# ---- Tab 2: Candlestick + Trade History ----
with chart_tab2:
    n_show = min(100, len(df_ohlc))
    ohlc_show = df_ohlc.tail(n_show).copy()
    
    fig_candle = go.Figure()
    
    # Add candlestick chart
    fig_candle.add_trace(go.Candlestick(
        x=ohlc_show["time"],
        open=ohlc_show["open"],
        high=ohlc_show["high"],
        low=ohlc_show["low"],
        close=ohlc_show["close"],
        name="OHLC",
        increasing_line_color="#1d9e75",
        decreasing_line_color="#e24b4a",
    ))
    
    # Add trade entry points
    if not df_trades.empty:
        recent_trades = df_trades.tail(n_trades_chart)
        
        # BUY entries
        buys = recent_trades[recent_trades["direction"] == "BUY"]
        if not buys.empty:
            fig_candle.add_trace(go.Scatter(
                x=buys["time"],
                y=buys["entry_price"],
                mode="markers",
                name="BUY Entry",
                marker=dict(symbol="triangle-up", size=12, color="#1d9e75", line=dict(width=2)),
            ))
            
            # Add SL and TP lines for BUY
            for idx, trade in buys.iterrows():
                # Stop Loss line
                fig_candle.add_shape(
                    type="line",
                    x0=trade["time"], x1=trade["time"],
                    y0=trade["sl_price"], y1=trade["entry_price"],
                    line=dict(color="#e24b4a", width=2, dash="dash"),
                )
                # Take Profit line
                fig_candle.add_shape(
                    type="line",
                    x0=trade["time"], x1=trade["time"],
                    y0=trade["tp_price"], y1=trade["entry_price"],
                    line=dict(color="#1d9e75", width=2, dash="dash"),
                )
        
        # SELL entries
        sells = recent_trades[recent_trades["direction"] == "SELL"]
        if not sells.empty:
            fig_candle.add_trace(go.Scatter(
                x=sells["time"],
                y=sells["entry_price"],
                mode="markers",
                name="SELL Entry",
                marker=dict(symbol="triangle-down", size=12, color="#e24b4a", line=dict(width=2)),
            ))
            
            # Add SL and TP lines for SELL
            for idx, trade in sells.iterrows():
                # Stop Loss line
                fig_candle.add_shape(
                    type="line",
                    x0=trade["time"], x1=trade["time"],
                    y0=trade["sl_price"], y1=trade["entry_price"],
                    line=dict(color="#e24b4a", width=2, dash="dash"),
                )
                # Take Profit line
                fig_candle.add_shape(
                    type="line",
                    x0=trade["time"], x1=trade["time"],
                    y0=trade["tp_price"], y1=trade["entry_price"],
                    line=dict(color="#1d9e75", width=2, dash="dash"),
                )
    
    fig_candle.update_layout(
        title=f"Candlestick Chart & Trade History — {selected_symbol}",
        yaxis_title="Price",
        xaxis_title="Time",
        height=500,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_candle, use_container_width=True)
    
    # Trade annotations
    if not df_trades.empty:
        recent_trades_display = df_trades.tail(n_trades_chart)[
            ["time", "direction", "entry_price", "sl_price", "tp_price", "pnl", "result", "confidence"]
        ].sort_values("time", ascending=False)
        
        st.subheader("Trade Details")
        st.dataframe(
            recent_trades_display,
            use_container_width=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry", format="%.5f"),
                "sl_price": st.column_config.NumberColumn("SL", format="%.5f"),
                "tp_price": st.column_config.NumberColumn("TP", format="%.5f"),
                "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            }
        )

# ---- Tab 3: Trade Distribution ----
with chart_tab3:
    if df_trades.empty:
        st.info("Belum ada data trade.")
    else:
        c1, c2 = st.columns(2)

        # PnL distribution
        with c1:
            fig_pnl = go.Figure()
            wins_pnl   = df_trades[df_trades["result"] == "WIN"]["pnl"]
            losses_pnl = df_trades[df_trades["result"] == "LOSS"]["pnl"]
            fig_pnl.add_trace(go.Histogram(x=wins_pnl,   name="Win",  marker_color="#1d9e75", nbinsx=20))
            fig_pnl.add_trace(go.Histogram(x=losses_pnl, name="Loss", marker_color="#e24b4a", nbinsx=20))
            fig_pnl.update_layout(
                title="Distribusi PnL per Trade ($)",
                barmode="overlay",
                height=320,
                xaxis_title="PnL ($)",
                yaxis_title="Frekuensi",
            )
            fig_pnl.update_traces(opacity=0.7)
            st.plotly_chart(fig_pnl, use_container_width=True)

        # Win/Loss pie
        with c2:
            win_count  = (df_trades["result"] == "WIN").sum()
            loss_count = (df_trades["result"] == "LOSS").sum()
            fig_pie = go.Figure(go.Pie(
                labels=["Win", "Loss"],
                values=[win_count, loss_count],
                marker_colors=["#1d9e75", "#e24b4a"],
                hole=0.4,
                textinfo="label+percent+value",
            ))
            fig_pie.update_layout(title="Rasio Win/Loss", height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Monthly PnL
        if "time" in df_trades.columns:
            df_monthly = df_trades.copy()
            df_monthly["month"] = pd.to_datetime(df_monthly["time"]).dt.to_period("M").astype(str)
            monthly = df_monthly.groupby("month")["pnl"].sum().reset_index()
            monthly["color"] = monthly["pnl"].apply(lambda x: "#1d9e75" if x > 0 else "#e24b4a")

            fig_monthly = go.Figure(go.Bar(
                x=monthly["month"],
                y=monthly["pnl"].round(2),
                marker_color=monthly["color"],
                name="Monthly PnL",
            ))
            fig_monthly.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_monthly.update_layout(
                title="PnL Bulanan ($)",
                height=280,
                yaxis_title="PnL ($)",
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        # Recent trades table
        st.subheader("📋 Trade Terbaru")
        display_cols = [c for c in ["time", "symbol", "direction", "pnl", "result", "confidence"] if c in df_trades.columns]
        st.dataframe(
            df_trades[display_cols].tail(20).sort_values("time", ascending=False),
            use_container_width=True,
            column_config={
                "pnl":        st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "result":     st.column_config.TextColumn("Hasil"),
            }
        )

# ---- Tab 4: Model Analysis ----
with chart_tab4:
    c1, c2 = st.columns(2)

    with c1:
        # Confidence distribution
        if not df_trades.empty and "confidence" in df_trades.columns:
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Histogram(
                x=df_trades[df_trades["result"]=="WIN"]["confidence"],
                name="Win", marker_color="#1d9e75", nbinsx=15, opacity=0.7,
            ))
            fig_conf.add_trace(go.Histogram(
                x=df_trades[df_trades["result"]=="LOSS"]["confidence"],
                name="Loss", marker_color="#e24b4a", nbinsx=15, opacity=0.7,
            ))
            fig_conf.update_layout(
                title="Distribusi Confidence Score",
                barmode="overlay", height=300,
                xaxis_title="Confidence", yaxis_title="Frekuensi",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    with c2:
        # Sinyal breakdown
        if not df_trades.empty and "direction" in df_trades.columns:
            sig_counts = df_trades["direction"].value_counts()
            fig_sig = go.Figure(go.Bar(
                x=sig_counts.index, y=sig_counts.values,
                marker_color=["#1d9e75" if s == "BUY" else "#e24b4a" for s in sig_counts.index],
            ))
            fig_sig.update_layout(
                title="Frekuensi Sinyal", height=300,
                yaxis_title="Jumlah",
            )
            st.plotly_chart(fig_sig, use_container_width=True)

    # Signal history (True vs Predicted) + Confusion Matrix
    st.subheader("📜 Signal History & Confusion Matrix")
    if df_trades.empty:
        st.info("Tidak ada data trade untuk menampilkan sejarah sinyal atau matriks kebingungan.")
    else:
        # Prepare predictions: prefer explicit prediction columns if available,
        # otherwise synthesize simple predicted labels from confidence.
        df_sig = df_trades.copy()
        # Prefer existing prediction columns if present (common names)
        pred_cols = ("pred_signal", "pred", "pred_label", "predicted")
        found = None
        for c in pred_cols:
            if c in df_sig.columns:
                found = c
                break
        if found is not None:
            df_sig["pred"] = df_sig[found]
        else:
            # Fallback synthesis: if confidence high, assume prediction==direction
            def _synth_pred(row):
                if row.get("confidence", 0) >= 0.6:
                    return row.get("direction")
                return np.random.choice(["BUY", "SELL", "HOLD"], p=[0.3, 0.3, 0.4])
            df_sig["pred"] = df_sig.apply(_synth_pred, axis=1)

        # Map signals to numeric series for plotting
        mapping = {"SELL": -1, "HOLD": 0, "BUY": 1}
        df_sig["true_num"] = df_sig["direction"].map(mapping).fillna(0)
        df_sig["pred_num"] = df_sig["pred"].map(mapping).fillna(0)

        # Signal history chart
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_sig["time"], y=df_sig["true_num"],
            mode="lines+markers", name="True Signal",
            line=dict(color="#3266ad"), marker=dict(size=6)
        ))
        fig_hist.add_trace(go.Scatter(
            x=df_sig["time"], y=df_sig["pred_num"],
            mode="lines+markers", name="Predicted Signal",
            line=dict(color="#e24b4a"), marker=dict(size=6)
        ))
        fig_hist.update_yaxes(tickvals=[-1, 0, 1], ticktext=["SELL", "HOLD", "BUY"])
        fig_hist.update_layout(title="Signal History (True vs Predicted)", height=300, hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Normalized confusion matrix (rows = true, cols = pred)
        labels = ["BUY", "SELL", "HOLD"]
        cm = pd.crosstab(df_sig["direction"], df_sig["pred"], rownames=["True"], colnames=["Pred"]) \
               .reindex(index=labels, columns=labels, fill_value=0)
        # normalize per-row (true class)
        cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_norm.values,
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0, zmax=1,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Value: %{z:.2f}<extra></extra>",
            text=cm_norm.round(2).values,
            texttemplate="%{text}",
        ))
        fig_cm.update_layout(title="Confusion Matrix (Normalized by True Class)", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Download buttons for confusion matrix
        try:
            cm_csv = cm.to_csv()
            cm_norm_csv = cm_norm.to_csv()
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download Confusion Matrix (counts)",
                    data=cm_csv,
                    file_name=f"confusion_matrix_{selected_symbol}.csv",
                    mime="text/csv",
                )
            with col_dl2:
                st.download_button(
                    "Download Confusion Matrix (normalized)",
                    data=cm_norm_csv,
                    file_name=f"confusion_matrix_normalized_{selected_symbol}.csv",
                    mime="text/csv",
                )
        except Exception:
            # Non-fatal: continue without download buttons
            pass

    # Model info
    if model_info:
        st.subheader("ℹ️ Informasi Model")
        st.json(model_info)
    else:
        st.info("Model info tersedia setelah API online.")

# ---- Tab 5: Trade History Table ----
with chart_tab5:
    if df_trades.empty:
        st.info("Belum ada data trade.")
    else:
        st.subheader(f"📋 Sejarah Trade Lengkap ({len(df_trades)} trades)")
        
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            filter_direction = st.multiselect("Filter Direction", ["BUY", "SELL"], default=["BUY", "SELL"])
        with col_filter2:
            filter_result = st.multiselect("Filter Result", ["WIN", "LOSS"], default=["WIN", "LOSS"])
        with col_filter3:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, step=0.05)
        
        # Apply filters
        df_filtered = df_trades.copy()
        df_filtered = df_filtered[df_filtered["direction"].isin(filter_direction)]
        df_filtered = df_filtered[df_filtered["result"].isin(filter_result)]
        df_filtered = df_filtered[df_filtered["confidence"] >= min_confidence]
        df_filtered = df_filtered.sort_values("time", ascending=False)
        
        # Display statistics for filtered data
        if not df_filtered.empty:
            stats_c1, stats_c2, stats_c3, stats_c4, stats_c5 = st.columns(5)
            with stats_c1:
                st.metric("Total Trades", len(df_filtered))
            with stats_c2:
                win_pct = (df_filtered["result"] == "WIN").sum() / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
                st.metric("Win Rate", f"{win_pct:.1f}%")
            with stats_c3:
                total_pnl = df_filtered["pnl"].sum()
                st.metric("Total PnL", f"${total_pnl:.2f}", delta=f"{(total_pnl/10000*100):.1f}% ROI")
            with stats_c4:
                avg_pnl = df_filtered["pnl"].mean()
                st.metric("Avg PnL/Trade", f"${avg_pnl:.2f}")
            with stats_c5:
                avg_conf = df_filtered["confidence"].mean()
                st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
            
            st.divider()
            
            # Detailed table
            display_cols = ["time", "direction", "entry_price", "sl_price", "tp_price", "pnl", "result", "confidence"]
            df_display = df_filtered[display_cols].copy()
            df_display["time"] = pd.to_datetime(df_display["time"]).dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(
                df_display,
                use_container_width=True,
                column_config={
                    "time": st.column_config.TextColumn("Waktu", width="medium"),
                    "direction": st.column_config.TextColumn("Arah"),
                    "entry_price": st.column_config.NumberColumn("Entry", format="%.5f"),
                    "sl_price": st.column_config.NumberColumn("SL", format="%.5f"),
                    "tp_price": st.column_config.NumberColumn("TP", format="%.5f"),
                    "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                    "result": st.column_config.TextColumn("Hasil"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                },
                hide_index=True,
            )
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History (CSV)",
                data=csv,
                file_name=f"trade_history_{selected_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# =====================================================================
# Auto refresh
# =====================================================================
if auto_refresh:
    time.sleep(30)
    st.rerun()
elif refresh_btn:
    st.rerun()

# Footer
st.divider()
st.caption(f"ML Forex Advisor Dashboard | {datetime.now().strftime('%d %b %Y %H:%M:%S')} | Demo mode: {demo_mode}")
