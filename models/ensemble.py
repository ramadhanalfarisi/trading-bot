"""
models/ensemble.py
Menggabungkan prediksi LSTM dan XGBoost menjadi satu sinyal akhir.
"""
import numpy as np
import joblib
from pathlib import Path
from loguru import logger

LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


class EnsemblePredictor:
    """
    Menggabungkan prediksi LSTM dan XGBoost dengan weighted soft voting.
    Output sinyal: HOLD / BUY / SELL + confidence score.
    """

    def __init__(self, config: dict):
        self.cfg = config["ensemble"]
        self.risk_cfg = config["risk"]
        self.model_path = Path(config["paths"]["ensemble_model"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.lstm_weight  = self.cfg["lstm_weight"]
        self.xgb_weight   = self.cfg["xgboost_weight"]
        self.conf_thresh  = self.cfg["confidence_threshold"]
        # optional tuning knobs to adjust HOLD bias at decision time
        self.hold_penalty = float(self.cfg.get("hold_penalty", 1.0))
        self.min_signal_prob = float(self.cfg.get("min_signal_prob", self.conf_thresh))
        self.meta = {}  # metadata tersimpan

    # ------------------------------------------------------------------
    # Prediksi gabungan
    # ------------------------------------------------------------------

    def predict(
        self,
        lstm_proba: np.ndarray,
        xgb_proba: np.ndarray,
    ) -> dict:
        """
        Gabungkan probabilitas dari kedua model.

        Args:
            lstm_proba: shape (n, 3) dari LSTMTrainer.predict_proba()
            xgb_proba:  shape (n, 3) dari XGBoostTrainer.predict_proba()

        Returns:
            dict dengan signal, confidence, dan detail per kelas
        """
        # Weighted average
        combined = self.lstm_weight * lstm_proba + self.xgb_weight * xgb_proba

        # Per-sample output
        results = []
        for i in range(len(combined)):
            proba = combined[i]
            # class probabilities
            hold_p = float(proba[0])
            buy_p = float(proba[1])
            sell_p = float(proba[2])

            # Decision rule: prefer BUY/SELL only if its prob exceeds both HOLD
            # and the confidence threshold. This reduces HOLD bias while
            # avoiding too many false signals.
            best_side = max(buy_p, sell_p)
            # apply hold penalty and configurable minimum signal probability
            effective_hold = hold_p * self.hold_penalty
            # if user set a positive min_signal_prob, use it as override;
            # otherwise fall back to configured confidence threshold
            min_required = self.min_signal_prob if self.min_signal_prob > 0 else self.conf_thresh
            if best_side > effective_hold and best_side >= min_required:
                pred_class = 1 if buy_p >= sell_p else 2
                confidence = best_side
            else:
                pred_class = 0
                confidence = hold_p

            results.append({
                "signal": LABEL_MAP[pred_class],
                "signal_id": pred_class,
                "confidence": round(confidence, 4),
                "proba_hold": round(hold_p, 4),
                "proba_buy":  round(buy_p, 4),
                "proba_sell": round(sell_p, 4),
            })

        return results if len(results) > 1 else results[0]

    def predict_single(
        self,
        lstm_proba: np.ndarray,
        xgb_proba: np.ndarray,
        atr_value: float,
        current_price: float,
        symbol_point: float = 0.00001,
    ) -> dict:
        """
        Prediksi untuk satu titik waktu + hitung SL/TP.

        Args:
            lstm_proba: shape (3,) atau (1,3)
            xgb_proba:  shape (3,) atau (1,3)
            atr_value:   nilai ATR saat ini
            current_price: harga close saat ini
            symbol_point: point size symbol

        Returns:
            dict lengkap dengan signal, confidence, sl_price, tp_price, sl_pips, tp_pips
        """
        if lstm_proba.ndim == 1:
            lstm_proba = lstm_proba[np.newaxis, :]
        if xgb_proba.ndim == 1:
            xgb_proba = xgb_proba[np.newaxis, :]

        result = self.predict(lstm_proba, xgb_proba)
        if isinstance(result, list):
            result = result[0]

        # Hitung SL/TP berbasis ATR
        sl_mult = float(self.risk_cfg.get("sl_atr_multiplier", 2.0))
        tp_mult = float(self.risk_cfg.get("tp_atr_multiplier", 3.0))
        sl_dist = atr_value * sl_mult
        tp_dist = atr_value * tp_mult

        if result["signal"] == "BUY":
            sl_price = round(current_price - sl_dist, 5)
            tp_price = round(current_price + tp_dist, 5)
        elif result["signal"] == "SELL":
            sl_price = round(current_price + sl_dist, 5)
            tp_price = round(current_price - tp_dist, 5)
        else:
            sl_price = tp_price = 0.0

        pips_factor = 1.0 / symbol_point / 10
        result.update({
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_pips": round(sl_dist / symbol_point / 10, 1) if symbol_point else 0,
            "tp_pips": round(tp_dist / symbol_point / 10, 1) if symbol_point else 0,
            "atr_value": round(atr_value, 5),
            "current_price": current_price,
        })
        return result

    # ------------------------------------------------------------------
    # Hitung lot size berdasarkan risiko
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        balance: float,
        sl_pips: float,
        pip_value: float = 10.0,
        volume_min: float = 0.01,
        volume_max: float = 10.0,
        volume_step: float = 0.01,
    ) -> float:
        """
        Hitung lot size berdasarkan persentase risiko balance.

        pip_value: nilai per pip per lot (USD) — default 10 untuk EURUSD 1 lot
        """
        risk_pct = self.risk_cfg["max_risk_per_trade"]
        risk_amount = balance * risk_pct
        if sl_pips <= 0:
            return volume_min
        raw_lot = risk_amount / (sl_pips * pip_value)
        # Bulatkan ke volume_step
        lot = round(round(raw_lot / volume_step) * volume_step, 2)
        lot = max(volume_min, min(lot, volume_max))
        return lot

    # ------------------------------------------------------------------
    # Simpan / muat metadata ensemble
    # ------------------------------------------------------------------

    def save(self, meta: dict = None):
        data = {"weights": {"lstm": self.lstm_weight, "xgb": self.xgb_weight},
                "conf_threshold": self.conf_thresh, "meta": meta or {}}
        joblib.dump(data, self.model_path)
        logger.info(f"💾 Ensemble config disimpan ke {self.model_path}")

    def load(self):
        if not self.model_path.exists():
            logger.warning("Ensemble config tidak ditemukan, menggunakan default.")
            return
        data = joblib.load(self.model_path)
        self.lstm_weight = data["weights"]["lstm"]
        self.xgb_weight  = data["weights"]["xgb"]
        self.conf_thresh  = data["conf_threshold"]
        self.meta = data.get("meta", {})
        logger.info(f"✅ Ensemble dimuat | LSTM w={self.lstm_weight} | XGB w={self.xgb_weight}")
