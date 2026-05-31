"""
data/feature_engineering.py
Menghasilkan fitur teknikal dari data OHLCV mentah.
"""
import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Membuat fitur teknikal dari data OHLCV."""

    def __init__(self, config: dict):
        self.cfg = config["features"]
        self.label_cfg = config["label"]

    # ------------------------------------------------------------------
    # Entry point utama
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame, add_labels: bool = True) -> pd.DataFrame:
        """
        Build semua fitur dari DataFrame OHLCV.

        Args:
            df: DataFrame dengan kolom [open, high, low, close, volume]
            add_labels: Jika True, tambahkan kolom target (BUY/SELL/HOLD)

        Returns:
            DataFrame dengan semua fitur dan label (jika diminta)
        """
        logger.info(f"Building features dari {len(df)} bar...")
        feat = df.copy()

        feat = self._add_returns(feat)
        feat = self._add_moving_averages(feat)
        feat = self._add_rsi(feat)
        feat = self._add_macd(feat)
        feat = self._add_bollinger_bands(feat)
        feat = self._add_atr(feat)
        feat = self._add_stochastic(feat)
        feat = self._add_volume_features(feat)
        feat = self._add_candlestick_patterns(feat)
        feat = self._add_lag_features(feat)
        feat = self._add_time_features(feat)
        feat = self._add_rolling_stats(feat)
        feat = self._add_price_levels(feat)

        # Defragment DataFrame before adding the label column
        feat = feat.copy()

        if add_labels:
            feat = self._add_labels(feat)

        # Hapus baris dengan NaN (akibat lookback indikator)
        before = len(feat)
        feat.dropna(inplace=True)
        logger.info(f"✅ Features siap: {len(feat)} bar ({before - len(feat)} baris dihapus karena NaN)")
        return feat

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["oc_range"] = (df["close"] - df["open"]) / df["open"]
        return df

    # ------------------------------------------------------------------
    # Moving Averages
    # ------------------------------------------------------------------

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.cfg["ema_periods"]:
            col = f"ema_{period}"
            df[col] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"close_vs_{col}"] = (df["close"] - df[col]) / df[col]

        for period in self.cfg["sma_periods"]:
            col = f"sma_{period}"
            df[col] = df["close"].rolling(period).mean()
            df[f"close_vs_{col}"] = (df["close"] - df[col]) / df[col]

        # Golden/Death cross (EMA 9 vs EMA 21)
        df["ema9_vs_ema21"] = df["ema_9"] - df["ema_21"]
        df["ema21_vs_ema50"] = df["ema_21"] - df["ema_50"]
        df["ema50_vs_ema200"] = df["ema_50"] - df["ema_200"]

        # Price position relative to MAs
        df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)
        df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(int)
        return df

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["rsi_period"]
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, adjust=True).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=True).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_change"] = df["rsi"].diff()
        return df

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.cfg["macd_fast"]
        slow = self.cfg["macd_slow"]
        signal = self.cfg["macd_signal"]

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_crossover"] = np.where(
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)), 1,
            np.where((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)), -1, 0)
        )
        df["macd_positive"] = (df["macd"] > 0).astype(int)
        return df

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["bb_period"]
        n_std = self.cfg["bb_std"]

        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()

        df["bb_upper"] = sma + n_std * std
        df["bb_lower"] = sma - n_std * std
        df["bb_mid"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).mean()).astype(int)
        return df

    # ------------------------------------------------------------------
    # ATR — Average True Range
    # ------------------------------------------------------------------

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["atr_period"]
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(com=period - 1, adjust=True).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        return df

    # ------------------------------------------------------------------
    # Stochastic Oscillator
    # ------------------------------------------------------------------

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        k = self.cfg["stoch_k"]
        d = self.cfg["stoch_d"]

        lowest_low = df["low"].rolling(k).min()
        highest_high = df["high"].rolling(k).max()
        df["stoch_k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(d).mean()
        df["stoch_crossover"] = np.where(
            (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)), 1,
            np.where((df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)), -1, 0)
        )
        return df

    # ------------------------------------------------------------------
    # Volume Features
    # ------------------------------------------------------------------

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-10)
        df["vol_change"] = df["volume"].pct_change()
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["obv_ema"] = df["obv"].ewm(span=21).mean()
        df["obv_vs_ema"] = df["obv"] - df["obv_ema"]
        return df

    # ------------------------------------------------------------------
    # Candlestick Patterns
    # ------------------------------------------------------------------

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        body = df["close"] - df["open"]
        body_abs = body.abs()
        candle_range = df["high"] - df["low"]
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

        # Doji
        df["doji"] = (body_abs / (candle_range + 1e-10) < 0.1).astype(int)
        # Hammer (lower shadow > 2× body, small upper shadow)
        df["hammer"] = (
            (lower_shadow > 2 * body_abs) &
            (upper_shadow < body_abs) &
            (body_abs > 0)
        ).astype(int)
        # Shooting Star
        df["shooting_star"] = (
            (upper_shadow > 2 * body_abs) &
            (lower_shadow < body_abs) &
            (body_abs > 0)
        ).astype(int)
        # Bullish/Bearish Engulfing
        prev_body = body.shift(1)
        df["bullish_engulf"] = (
            (body > 0) & (prev_body < 0) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)
        df["bearish_engulf"] = (
            (body < 0) & (prev_body > 0) &
            (df["close"] < df["open"].shift(1)) &
            (df["open"] > df["close"].shift(1))
        ).astype(int)
        # Body ratio
        df["body_ratio"] = body_abs / (candle_range + 1e-10)
        df["candle_direction"] = np.sign(body).astype(int)
        return df

    # ------------------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.cfg["lag_returns"]:
            df[f"lag_return_{lag}"] = df["log_return"].shift(lag)
            df[f"lag_rsi_{lag}"] = df["rsi"].shift(lag)
            df[f"lag_macd_hist_{lag}"] = df["macd_hist"].shift(lag)
        return df

    # ------------------------------------------------------------------
    # Time Features (dari index datetime)
    # ------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if hasattr(idx, "hour"):
            df["hour"] = idx.hour
            df["day_of_week"] = idx.dayofweek
            # Sesi trading (UTC)
            df["session_asian"] = ((idx.hour >= 0) & (idx.hour < 8)).astype(int)
            df["session_london"] = ((idx.hour >= 8) & (idx.hour < 16)).astype(int)
            df["session_newyork"] = ((idx.hour >= 13) & (idx.hour < 21)).astype(int)
            df["session_overlap"] = ((idx.hour >= 13) & (idx.hour < 16)).astype(int)
        return df

    # ------------------------------------------------------------------
    # Rolling Statistics
    # ------------------------------------------------------------------

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.cfg["rolling_windows"]:
            df[f"roll_mean_{w}"] = df["log_return"].rolling(w).mean()
            df[f"roll_std_{w}"] = df["log_return"].rolling(w).std()
            df[f"roll_skew_{w}"] = df["log_return"].rolling(w).skew()
            df[f"roll_max_{w}"] = df["high"].rolling(w).max()
            df[f"roll_min_{w}"] = df["low"].rolling(w).min()
        return df

    # ------------------------------------------------------------------
    # Price Levels (Support/Resistance sederhana)
    # ------------------------------------------------------------------

    def _add_price_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        w = 20
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["resist1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["support1"] = 2 * df["pivot"] - df["high"].shift(1)
        df["price_vs_pivot"] = (df["close"] - df["pivot"]) / df["pivot"]
        # 20-bar swing highs/lows
        df["swing_high"] = df["high"].rolling(w).max()
        df["swing_low"] = df["low"].rolling(w).min()
        df["range_position"] = (df["close"] - df["swing_low"]) / (df["swing_high"] - df["swing_low"] + 1e-10)
        return df

    # ------------------------------------------------------------------
    # Label / Target
    # ------------------------------------------------------------------

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Buat label: 0=HOLD, 1=BUY, 2=SELL
        Berdasarkan pergerakan harga N bar ke depan.
        """
        n = self.label_cfg["lookahead_bars"]
        pip_size = self.label_cfg.get("pip_size")
        if pip_size is None:
            pip_size = 0.01 if df["close"].mean() > 3 else 0.0001
        thresh_pct = self.label_cfg["threshold_pips"] * pip_size / df["close"].mean()
        logger.debug(f"Label threshold: {self.label_cfg['threshold_pips']} pips -> {thresh_pct:.6f} relative")

        future_return = df["close"].shift(-n) / df["close"] - 1

        df["label"] = 0  # HOLD default
        df.loc[future_return > thresh_pct, "label"] = 1    # BUY
        df.loc[future_return < -thresh_pct, "label"] = 2   # SELL

        # Hapus N baris terakhir (tidak ada future data)
        df = df.iloc[:-n]

        dist = df["label"].value_counts().sort_index()
        logger.info(f"Label distribution: HOLD={dist.get(0,0)} | BUY={dist.get(1,0)} | SELL={dist.get(2,0)}")
        return df

    # ------------------------------------------------------------------
    # Ambil daftar nama fitur (tanpa label & OHLCV mentah)
    # ------------------------------------------------------------------

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        exclude = ["open", "high", "low", "close", "volume", "label",
                   "ema_9", "ema_21", "ema_50", "ema_200",
                   "sma_20", "sma_50", "bb_upper", "bb_lower", "bb_mid",
                   "pivot", "resist1", "support1", "swing_high", "swing_low"]
        return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Test dengan data dummy
    np.random.seed(42)
    n = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.randn(n) * 0.0005)
    noise = np.abs(np.random.randn(n) * 0.0002)
    df_raw = pd.DataFrame({
        "open": close - np.random.randn(n) * 0.0001,
        "high": close + noise,
        "low": close - noise,
        "close": close,
        "volume": np.random.randint(100, 5000, n).astype(float),
    }, index=idx)

    fe = FeatureEngineer(cfg)
    df_feat = fe.build_features(df_raw)
    feat_cols = fe.get_feature_columns(df_feat)
    print(f"Total fitur: {len(feat_cols)}")
    print(df_feat[feat_cols].tail(3))
