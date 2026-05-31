"""
models/tick_backtest.py
Replay tick history untuk evaluasi model ML pada data tick nyata.
"""
import numpy as np
import pandas as pd
from loguru import logger

from models.backtest import Backtester

TIMEFRAME_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
}


class TickBacktester:
    """Replay tick history dan hitung metrik trading tick-level dengan leverage support."""

    def __init__(self, config: dict):
        self.cfg = config
        self.initial_balance = config.get("backtest", {}).get("initial_balance", 10_000.0)
        self.leverage = config.get("backtest", {}).get("leverage", 1)
        self.pip_value = config.get("backtest", {}).get("pip_value", 10.0)
        self.commission = config.get("backtest", {}).get("commission_per_trade", 0.0)
        self.point = config["data"].get("point", 0.00001)
        self.max_spread_pips = config["risk"].get("max_spread_pips", 3.0)
        self.use_leverage_for_sizing = config.get("backtest", {}).get("use_leverage_for_lot_sizing", True)
        
        # Leverage info
        self.margin_requirement = 1.0 / self.leverage  # e.g., 1:500 = 0.002 (0.2%)
        logger.info(f"TickBacktester initialized dengan leverage 1:{self.leverage} (margin req: {self.margin_requirement*100:.2f}%)")

    def _parse_timeframe(self, timeframe: str) -> str:
        return TIMEFRAME_MAP.get(timeframe.upper(), timeframe)

    def aggregate_ticks(self, df_ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        df = df_ticks.copy()
        df = df.sort_index()
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2.0

        tf = self._parse_timeframe(timeframe)
        bars = df["mid"].resample(tf).ohlc()
        bars["volume"] = df["bid"].resample(tf).count().fillna(0).astype(int)
        bars = bars.dropna()
        return bars

    def _calc_pnl(self, trade: dict, exit_price: float) -> float:
        """Hitung PnL dengan mempertimbangkan commission."""
        pips = (exit_price - trade["entry_price"]) / self.point / 10.0
        if trade["direction"] == "SELL":
            pips = -pips
        gross_pnl = pips * self.pip_value * trade["lot"]
        net_pnl = gross_pnl - self.commission
        return net_pnl

    def _calc_lot(self, balance: float, sl_price: float, entry_price: float) -> float:
        """
        Hitung lot size dengan mempertimbangkan leverage.
        
        Args:
            balance: Available balance
            sl_price: Stop loss price
            entry_price: Entry price
            
        Returns:
            Lot size yang memenuhi risk management dan margin requirement
        """
        sl_dist = abs(sl_price - entry_price)
        sl_pips = sl_dist / self.point / 10.0
        risk_amount = balance * self.cfg["risk"]["max_risk_per_trade"]
        
        if sl_pips <= 0:
            return 0.01
        
        # Base lot size based on risk
        raw_lot = risk_amount / (sl_pips * self.pip_value)
        
        # Calculate max lot based on available margin with leverage
        if self.use_leverage_for_sizing and self.leverage > 1:
            # Available margin = balance * leverage
            available_margin = balance * self.leverage
            # Max lot based on margin (1 standard lot = 100,000 units, needs $1000 margin at 1:100)
            # For EURUSD at current price ~1.08, 1 lot = 100,000 EUR = ~108,000 USD
            # Margin needed = position_size * price * margin_requirement
            # lot_value = lot * 100_000 * entry_price
            margin_needed_per_lot = 100_000 * entry_price * self.margin_requirement
            max_lot_by_margin = available_margin / margin_needed_per_lot if margin_needed_per_lot > 0 else 100.0
            
            # Take minimum of risk-based and margin-based
            raw_lot = min(raw_lot, max_lot_by_margin)
        
        # Cap the lot size (max 10 standard lots for safety)
        lot = max(0.01, min(round(raw_lot, 2), 10.0))
        return lot

    def _calc_metrics(self, trades: list, equity_curve: list) -> dict:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_pnl": 0,
                "final_balance": self.initial_balance,
                "return_pct": 0,
                "avg_pnl": 0,
                "avg_trade_duration": 0,
            }

        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_pnl = float(np.sum(pnls))
        win_rate = float(len(wins) / len(pnls))
        gross_profit = float(np.sum(wins))
        gross_loss = float(abs(np.sum(losses)))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        eq = np.array(equity_curve)
        returns = np.diff(eq) / (eq[:-1] + 1e-9)
        sharpe = float((returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 24)) if len(returns) > 1 else 0.0

        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / (peak + 1e-9)
        max_drawdown = float(np.min(drawdowns))

        durations = [((t["exit_time"] - t["entry_time"]).total_seconds() / 60.0) for t in trades]
        avg_duration = float(np.mean(durations)) if durations else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(self.initial_balance + total_pnl, 2),
            "return_pct": round(total_pnl / self.initial_balance * 100, 2),
            "avg_pnl": round(float(np.mean(pnls)), 2),
            "avg_trade_duration": round(avg_duration, 2),
            "leverage": self.leverage,
            "margin_requirement_pct": round(self.margin_requirement * 100, 2),
            "initial_balance": self.initial_balance,
            "commission_total": round(self.commission * len(trades), 2),
        }

    def run(
        self,
        df_ticks: pd.DataFrame,
        symbol: str,
        predict_fn,
        timeframe: str = "M1",
        verbose: bool = True,
    ) -> dict:
        df_ticks = df_ticks.copy()
        df_ticks = df_ticks.sort_index()
        if df_ticks.empty:
            logger.warning("Tick history kosong. Tidak ada yang bisa diuji.")
            return self._calc_metrics([], [self.initial_balance])

        df_ticks["mid"] = (df_ticks["bid"] + df_ticks["ask"]) / 2.0
        df_bars = self.aggregate_ticks(df_ticks, timeframe)
        logger.info(f"TickBacktester: {len(df_ticks)} ticks -> {len(df_bars)} bars saat agregasi {timeframe}.")
        if df_bars.empty:
            logger.warning("Tidak ada bar yang terbentuk dari tick history.")
            return self._calc_metrics([], [self.initial_balance])

        min_bars = max(
            max(self.cfg["features"]["ema_periods"]),
            max(self.cfg["features"]["sma_periods"]),
            self.cfg["features"]["bb_period"],
            self.cfg["features"]["atr_period"],
            max(self.cfg["features"]["lag_returns"]) + 1,
            self.cfg["label"]["lookahead_bars"] + 1,
        )
        if len(df_bars) < min_bars:
            logger.warning(
                f"Data bar terlalu singkat untuk membangun fitur ML. "
                f"Bar tersedia: {len(df_bars)}; minimum estimasi: {min_bars}. "
                f"Naikkan jumlah tick atau gunakan timeframe yang lebih kecil."
            )

        indicator_period = pd.Timedelta(self._parse_timeframe(timeframe))
        trades = []
        balance = self.initial_balance
        equity_curve = [balance]
        open_trade = None
        last_tick_pos = 0
        bar_times = list(df_bars.index)

        for i, bar_time in enumerate(bar_times):
            result = predict_fn(df_bars.iloc[: i + 1])
            if isinstance(result, dict):
                signal = result.get("signal", "HOLD")
                confidence = result.get("confidence", 0.0)
            else:
                signal = "HOLD"
                confidence = 0.0
            entry_time = bar_time + indicator_period
            tick_slice = df_ticks.loc[df_ticks.index >= entry_time]
            if tick_slice.empty:
                continue
            # Close existing trade if price hits SL/TP before next bar
            if open_trade is not None:
                for tick_idx in range(last_tick_pos, len(df_ticks)):
                    tick = df_ticks.iloc[tick_idx]
                    if tick.name < open_trade["entry_time"]:
                        continue
                    hit, pnl = self._check_tick_exit(open_trade, tick)
                    if hit:
                        balance += pnl
                        open_trade["exit_time"] = tick.name
                        open_trade["exit_price"] = tick["bid"] if open_trade["direction"] == "BUY" else tick["ask"]
                        open_trade["pnl"] = pnl
                        open_trade["result"] = "WIN" if pnl > 0 else "LOSS"
                        trades.append(open_trade)
                        open_trade = None
                        last_tick_pos = tick_idx + 1
                        equity_curve.append(balance)
                        break
                if open_trade is not None:
                    last_tick_pos = len(df_ticks)

            if open_trade is None and signal in ("BUY", "SELL"):
                entry_tick = tick_slice.iloc[0]
                sl_price, tp_price = self._resolve_sl_tp(signal, entry_tick["mid"], df_bars.iloc[i]["close"])
                lot = self._calc_lot(balance, sl_price, float(entry_tick["mid"]))
                open_trade = {
                    "direction": signal,
                    "entry_time": entry_tick.name,
                    "entry_price": float(entry_tick["ask"] if signal == "BUY" else entry_tick["bid"]),
                    "sl": sl_price,
                    "tp": tp_price,
                    "lot": lot,
                    "signal_confidence": confidence,
                }
                open_trade["symbol"] = symbol
                # Robustly determine the integer position of the entry tick in case
                # the index contains duplicate timestamps (get_loc may return a slice).
                try:
                    loc = df_ticks.index.get_loc(entry_tick.name)
                    if isinstance(loc, slice):
                        # slice -> take left insertion position
                        last_tick_pos = int(df_ticks.index.searchsorted(entry_tick.name, side="left"))
                    elif hasattr(loc, "__len__") and not isinstance(loc, (int, np.integer)):
                        # list/array of positions -> take the last one + 1
                        last_tick_pos = int(loc[-1]) + 1
                    else:
                        # single integer position
                        last_tick_pos = int(loc) + 1
                except Exception:
                    # Fallback to searchsorted which always returns a numeric insertion index
                    last_tick_pos = int(df_ticks.index.searchsorted(entry_tick.name, side="left"))

        # Tutup trade terakhir jika masih terbuka
        if open_trade is not None:
            last_tick = df_ticks.iloc[-1]
            exit_price = float(last_tick["bid"] if open_trade["direction"] == "BUY" else last_tick["ask"])
            pnl = self._calc_pnl(open_trade, exit_price)
            balance += pnl
            open_trade["exit_time"] = last_tick.name
            open_trade["exit_price"] = exit_price
            open_trade["pnl"] = pnl
            open_trade["result"] = "WIN" if pnl > 0 else "LOSS"
            trades.append(open_trade)
            equity_curve.append(balance)

        metrics = self._calc_metrics(trades, equity_curve)
        metrics["trade_count"] = len(trades)
        metrics["symbol"] = symbol
        return {"metrics": metrics, "trades": trades, "bars": df_bars}

    def _resolve_sl_tp(self, direction: str, price: float, close_price: float) -> tuple:
        atr = abs(close_price * 0.001)
        sl_dist = atr * self.cfg["risk"]["sl_atr_multiplier"]
        tp_dist = atr * self.cfg["risk"]["tp_atr_multiplier"]
        if direction == "BUY":
            return round(price - sl_dist, 5), round(price + tp_dist, 5)
        return round(price + sl_dist, 5), round(price - tp_dist, 5)

    def _check_tick_exit(self, trade: dict, tick: pd.Series) -> tuple:
        if trade["direction"] == "BUY":
            if tick["bid"] <= trade["sl"]:
                return True, self._calc_pnl(trade, trade["sl"])
            if tick["bid"] >= trade["tp"]:
                return True, self._calc_pnl(trade, trade["tp"])
        if trade["direction"] == "SELL":
            if tick["ask"] >= trade["sl"]:
                return True, self._calc_pnl(trade, trade["sl"])
            if tick["ask"] <= trade["tp"]:
                return True, self._calc_pnl(trade, trade["tp"])
        return False, 0.0
