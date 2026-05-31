"""
api/predictor.py
Inference engine untuk prediksi real-time dari data MT5.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from data.collector import MT5Collector
from data.feature_engineering import FeatureEngineer
from data.preprocessor import Preprocessor
from models.lstm_model import LSTMTrainer
from models.xgboost_model import XGBoostTrainer
from models.ensemble import EnsemblePredictor


class ForexPredictor:
    """
    Menghasilkan prediksi sinyal trading secara real-time.
    Dipanggil oleh Flask API maupun langsung dari kode lain.
    """

    def __init__(self, config: dict):
        self.config = config
        self.risk_cfg = config["risk"]

        self.collector    = MT5Collector(config)
        self.feat_eng     = FeatureEngineer(config)
        self.preprocessor = Preprocessor(config)
        self.lstm_trainer = LSTMTrainer(config)
        self.xgb_trainer  = XGBoostTrainer(config)
        self.ensemble     = EnsemblePredictor(config)

        self._loaded = False

    # ------------------------------------------------------------------
    # Muat semua model
    # ------------------------------------------------------------------

    def load_models(self):
        """Muat semua model yang sudah ditraining."""
        self.preprocessor.load_scaler()
        self.xgb_trainer.load_model()

        # LSTM opsional (skip jika file tidak ada)
        try:
            seq_len   = self.config["lstm"]["sequence_length"]
            n_features = len(self.preprocessor.feature_cols)
            self.lstm_trainer.build_model(n_features)
            self.lstm_trainer.load_model()
            self.use_lstm = True
        except FileNotFoundError:
            logger.warning("LSTM model tidak ditemukan — hanya XGBoost yang digunakan.")
            self.use_lstm = False

        self.ensemble.load()
        self._loaded = True
        logger.info("✅ Semua model berhasil dimuat.")

    # ------------------------------------------------------------------
    # Prediksi dari DataFrame OHLCV langsung
    # ------------------------------------------------------------------

    def predict_from_ohlcv(
        self,
        df_ohlcv: pd.DataFrame,
        symbol: str = "EURUSD",
    ) -> dict:
        """
        Ambil sinyal dari DataFrame OHLCV.

        Args:
            df_ohlcv: DataFrame [open, high, low, close, volume]
            symbol: nama symbol untuk info point

        Returns:
            dict lengkap dengan signal, confidence, sl, tp, lot, dll
        """
        if not self._loaded:
            self.load_models()

        # Build fitur
        df_feat = self.feat_eng.build_features(df_ohlcv, add_labels=False)
        feat_cols = self.preprocessor.feature_cols

        if df_feat.empty or not feat_cols:
            required_bars = max(
                self.feat_eng.cfg.get("ema_periods", [0]) +
                self.feat_eng.cfg.get("sma_periods", [0]) +
                [self.feat_eng.cfg.get("bb_period", 0), self.feat_eng.cfg.get("atr_period", 0)] +
                [max(self.feat_eng.cfg.get("lag_returns", [0])) + 1, self.config["label"]["lookahead_bars"] + 1]
            )
            raise ValueError(
                f"Insufficient data to compute features for prediction. "
                f"Input OHLCV bars: {len(df_ohlcv)}; estimated minimum bars required: {required_bars}. "
                f"Gunakan lebih banyak tick/bar atau timeframe lebih kecil."
            )

        # Ambil baris terakhir saja untuk prediksi
        df_last = df_feat[feat_cols].iloc[-1:]
        if df_last.empty:
            raise ValueError(
                f"Tidak cukup bar untuk menghasilkan fitur prediksi. "
                f"OHLCV bar tersedia: {len(df_ohlcv)}."
            )

        X_scaled = self.preprocessor.transform(df_last)
        if X_scaled.shape[0] == 0:
            raise ValueError("Transformasi fitur menghasilkan 0 sampel — data terlalu sedikit.")

        # XGBoost prediksi
        xgb_proba = self.xgb_trainer.predict_proba(X_scaled)  # (1, 3)

        # LSTM prediksi (perlu sequence)
        if self.use_lstm:
            seq_len = self.config["lstm"]["sequence_length"]
            n_feat  = X_scaled.shape[1]

            # Ambil seq_len baris terakhir
            if len(df_feat) >= seq_len:
                X_all  = self.preprocessor.transform(df_feat[feat_cols])
                X_seq  = X_all[-seq_len:][np.newaxis, :, :]  # (1, seq, feat)
                lstm_proba = self.lstm_trainer.predict_proba(X_seq)  # (1, 3)
            else:
                logger.warning(f"Data kurang dari {seq_len} bar — LSTM menggunakan zero padding.")
                lstm_proba = xgb_proba  # fallback ke XGBoost

            lstm_p = lstm_proba[0]
        else:
            lstm_p = xgb_proba[0]

        xgb_p = xgb_proba[0]

        # Ambil info teknikal terkini
        last_row   = df_feat.iloc[-1]
        atr_value  = float(last_row.get("atr", df_ohlcv["close"].iloc[-1] * 0.001))
        close_price = float(df_ohlcv["close"].iloc[-1])

        # Info symbol
        sym_info = self.collector.get_symbol_info(symbol) if self.collector.connected else {
            "point": 0.00001, "volume_min": 0.01, "volume_max": 10.0, "volume_step": 0.01
        }
        account  = self.collector.get_account_info()

        # Ensemble prediction + SL/TP
        result = self.ensemble.predict_single(
            lstm_proba=lstm_p,
            xgb_proba=xgb_p,
            atr_value=atr_value,
            current_price=close_price,
            symbol_point=sym_info["point"],
        )

        # Hitung lot size
        lot = self.ensemble.calculate_lot_size(
            balance=account["balance"],
            sl_pips=result["sl_pips"],
            volume_min=sym_info["volume_min"],
            volume_max=sym_info["volume_max"],
            volume_step=sym_info["volume_step"],
        )
        result["lot_size"] = lot
        result["symbol"] = symbol
        result["balance"] = account["balance"]

        # Risk filter check
        result["risk_passed"] = self._check_risk(result, account, symbol)

        logger.info(
            f"📡 {symbol} | {result['signal']} | conf={result['confidence']:.3f} | "
            f"SL={result['sl_price']} TP={result['tp_price']} | lot={lot} | "
            f"risk_ok={result['risk_passed']}"
        )
        return result

    # ------------------------------------------------------------------
    # Ambil data live dan prediksi langsung dari MT5
    # ------------------------------------------------------------------

    def predict_live(self, symbol: str = None, timeframe: str = None) -> dict:
        """
        Ambil data terbaru dari MT5 dan hasilkan prediksi.
        """
        symbol    = symbol    or self.config["data"]["primary_symbol"]
        timeframe = timeframe or self.config["data"]["timeframes"]["primary"]
        n_bars    = self.config["data"]["bars_inference"]

        if not self.collector.connected:
            self.collector.connect()

        df_ohlcv = self.collector.get_ohlcv(symbol, timeframe, n_bars, save=False)
        return self.predict_from_ohlcv(df_ohlcv, symbol)

    # ------------------------------------------------------------------
    # Risk Management Filter
    # ------------------------------------------------------------------

    def _check_risk(self, result: dict, account: dict, symbol: str) -> bool:
        """
        Cek apakah sinyal memenuhi semua filter risiko.
        Kembalikan True jika boleh dieksekusi.
        """
        reasons = []

        # 1. Confidence threshold
        if result["signal"] != "HOLD" and result["confidence"] < self.config["ensemble"]["confidence_threshold"]:
            reasons.append(f"confidence {result['confidence']:.3f} < {self.config['ensemble']['confidence_threshold']}")

        # 2. Max open trades
        if self.collector.connected:
            open_pos = self.collector.get_open_positions(symbol)
            if len(open_pos) >= self.risk_cfg["max_open_trades"]:
                reasons.append(f"max open trades {self.risk_cfg['max_open_trades']} tercapai")

        # 3. Spread check
        if self.collector.connected and result["signal"] != "HOLD":
            tick = self.collector.get_latest_tick(symbol)
            spread_pips = (tick["ask"] - tick["bid"]) / 0.00001 / 10
            if spread_pips > self.risk_cfg["max_spread_pips"]:
                reasons.append(f"spread {spread_pips:.1f} pips > {self.risk_cfg['max_spread_pips']}")

        # 4. Daily drawdown
        if account.get("equity") and account.get("balance"):
            dd = (account["balance"] - account["equity"]) / account["balance"]
            if dd > self.risk_cfg["max_daily_drawdown"]:
                reasons.append(f"drawdown {dd*100:.1f}% > {self.risk_cfg['max_daily_drawdown']*100:.0f}%")

        if reasons:
            logger.warning(f"⚠️ Risk filter GAGAL: {'; '.join(reasons)}")
            return False
        return True
