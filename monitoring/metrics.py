"""
monitoring/metrics.py
Pelacakan metrik performa model secara berkala,
deteksi model drift, dan sistem alert.
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


# ======================================================================
# MetricsTracker — catat & baca histori performa
# ======================================================================

class MetricsTracker:
    """
    Menyimpan histori metrik model ke file JSON lokal.
    Setiap sesi training/inference menulis satu record.
    """

    def __init__(self, config: dict):
        self.config = config
        self.log_dir = Path(config["paths"]["logs_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics_history.json"
        self.trades_file  = self.log_dir / "trades_log.json"

    # ------------------------------------------------------------------
    # Simpan metrik training
    # ------------------------------------------------------------------

    def record_training(self, symbol: str, metrics: dict):
        """
        Catat hasil training ke histori.
        metrics harus berisi: val_accuracy, sharpe_ratio, win_rate,
        max_drawdown, profit_factor, total_trades, dll
        """
        record = {
            "ts":        datetime.utcnow().isoformat(),
            "symbol":    symbol,
            "type":      "training",
            **metrics,
        }
        self._append(self.metrics_file, record)
        logger.info(f"📝 Metrik training disimpan: acc={metrics.get('val_accuracy','?')}")

    # ------------------------------------------------------------------
    # Simpan log trade dari EA
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict):
        """
        Catat satu trade yang dieksekusi.
        trade harus berisi: symbol, direction, entry_price, exit_price,
        pnl, result (WIN/LOSS), confidence, lot, timestamp
        """
        record = {"ts": datetime.utcnow().isoformat(), **trade}
        self._append(self.trades_file, record)

    # ------------------------------------------------------------------
    # Baca histori metrik
    # ------------------------------------------------------------------

    def load_metrics_history(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        records = self._read_all(self.metrics_file)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"])
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["ts"] >= cutoff]
        if symbol:
            df = df[df["symbol"] == symbol]
        return df.sort_values("ts")

    def load_trades(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        records = self._read_all(self.trades_file)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"])
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["ts"] >= cutoff]
        if symbol:
            df = df[df["symbol"] == symbol]
        return df.sort_values("ts")

    # ------------------------------------------------------------------
    # Hitung ringkasan performa live trading
    # ------------------------------------------------------------------

    def compute_live_performance(self, symbol: str = None) -> dict:
        """Hitung metrik dari trade log aktual."""
        df = self.load_trades(symbol)
        if df.empty:
            return {}

        wins   = df[df["result"] == "WIN"]
        losses = df[df["result"] == "LOSS"]
        pnls   = df["pnl"].tolist()

        gross_profit = wins["pnl"].sum() if len(wins) else 0
        gross_loss   = abs(losses["pnl"].sum()) if len(losses) else 1e-9
        total_pnl    = sum(pnls)

        # Equity curve untuk Sharpe & drawdown
        balance = 10000.0
        equity_curve = [balance]
        for p in pnls:
            balance += p
            equity_curve.append(balance)

        eq = np.array(equity_curve)
        rets = np.diff(eq) / (eq[:-1] + 1e-10)
        sharpe = float(rets.mean() / (rets.std() + 1e-10)) * np.sqrt(252 * 24)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak) / (peak + 1e-10)).min())

        return {
            "total_trades":  len(df),
            "win_trades":    len(wins),
            "loss_trades":   len(losses),
            "win_rate":      round(len(wins) / len(df), 4),
            "profit_factor": round(gross_profit / gross_loss, 3),
            "sharpe_ratio":  round(sharpe, 3),
            "max_drawdown":  round(max_dd, 4),
            "total_pnl":     round(total_pnl, 2),
            "final_balance": round(balance, 2),
            "avg_confidence": round(df["confidence"].mean(), 3) if "confidence" in df else None,
        }

    # ------------------------------------------------------------------
    # Helper I/O
    # ------------------------------------------------------------------

    def _append(self, path: Path, record: dict):
        records = self._read_all(path)
        records.append(record)
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def _read_all(self, path: Path) -> list:
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []


# ======================================================================
# DriftDetector — deteksi model drift
# ======================================================================

class DriftDetector:
    """
    Mendeteksi apakah performa model mengalami degradasi
    dibandingkan baseline training (model drift).

    Menggunakan metode sliding window: bandingkan akurasi
    N trade terbaru dengan akurasi baseline saat training.
    """

    def __init__(self, config: dict):
        self.threshold  = config["monitoring"]["drift_threshold"]     # default 0.10
        self.min_trades = 30   # minimal trade sebelum drift check
        self.window     = 50   # ukuran window evaluasi

    def check(self, trades_df: pd.DataFrame, baseline_accuracy: float) -> dict:
        """
        Cek drift pada DataFrame trades.

        Returns:
            dict dengan keys: drifted (bool), delta, window_accuracy, baseline_accuracy
        """
        if trades_df.empty or len(trades_df) < self.min_trades:
            return {"drifted": False, "reason": "data tidak cukup",
                    "window_accuracy": None, "baseline_accuracy": baseline_accuracy}

        recent = trades_df.tail(self.window)
        window_acc = (recent["result"] == "WIN").mean()
        delta = baseline_accuracy - window_acc

        drifted = delta > self.threshold

        result = {
            "drifted":           drifted,
            "window_accuracy":   round(float(window_acc), 4),
            "baseline_accuracy": round(float(baseline_accuracy), 4),
            "delta":             round(float(delta), 4),
            "window_size":       len(recent),
            "threshold":         self.threshold,
        }

        if drifted:
            logger.warning(
                f"⚠️  MODEL DRIFT TERDETEKSI! "
                f"Baseline: {baseline_accuracy:.3f} | Window: {window_acc:.3f} | "
                f"Delta: {delta:.3f} > threshold {self.threshold}"
            )
        else:
            logger.info(
                f"✅ Drift check OK | Window acc: {window_acc:.3f} | "
                f"Delta: {delta:.3f}"
            )
        return result

    def check_distribution_shift(self, recent_returns: np.ndarray,
                                  baseline_returns: np.ndarray) -> dict:
        """
        Cek pergeseran distribusi return menggunakan KS-test sederhana.
        Tidak memerlukan scipy — implementasi manual.
        """
        if len(recent_returns) < 10 or len(baseline_returns) < 10:
            return {"shifted": False, "reason": "data tidak cukup"}

        # Perbandingan statistik sederhana
        rec_mean = float(np.mean(recent_returns))
        rec_std  = float(np.std(recent_returns))
        base_mean = float(np.mean(baseline_returns))
        base_std  = float(np.std(baseline_returns))

        mean_shift = abs(rec_mean - base_mean)
        std_shift  = abs(rec_std  - base_std)

        shifted = mean_shift > 0.02 or std_shift > 0.015

        return {
            "shifted":    shifted,
            "rec_mean":   round(rec_mean, 5),
            "base_mean":  round(base_mean, 5),
            "rec_std":    round(rec_std, 5),
            "base_std":   round(base_std, 5),
            "mean_shift": round(mean_shift, 5),
            "std_shift":  round(std_shift, 5),
        }


# ======================================================================
# AlertManager — kirim notifikasi
# ======================================================================

class AlertManager:
    """
    Mengirim alert ke berbagai channel ketika kondisi tertentu terpenuhi.
    Saat ini mendukung: log file, Telegram (opsional), email (opsional).
    """

    def __init__(self, config: dict):
        self.config   = config
        self.mon_cfg  = config["monitoring"]
        self.log_dir  = Path(config["paths"]["logs_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log = self.log_dir / "alerts.json"
        self._alerts_sent: list = []

    # ------------------------------------------------------------------
    # Pemeriksaan threshold otomatis
    # ------------------------------------------------------------------

    def check_all(self, metrics: dict, drift_result: dict = None) -> list:
        """
        Periksa semua kondisi alert dari dict metrik.
        Kembalikan list alert yang ter-trigger.
        """
        triggered = []

        # Win rate rendah
        wr = metrics.get("win_rate", 1.0)
        if wr < self.mon_cfg["min_win_rate"]:
            triggered.append(self._make_alert(
                level="WARNING",
                code="LOW_WIN_RATE",
                msg=f"Win rate {wr*100:.1f}% di bawah threshold {self.mon_cfg['min_win_rate']*100:.0f}%",
                value=wr,
            ))

        # Sharpe rendah
        sr = metrics.get("sharpe_ratio", 99.0)
        if sr < self.mon_cfg["min_sharpe"]:
            triggered.append(self._make_alert(
                level="WARNING",
                code="LOW_SHARPE",
                msg=f"Sharpe ratio {sr:.2f} di bawah threshold {self.mon_cfg['min_sharpe']}",
                value=sr,
            ))

        # Drawdown tinggi
        dd = metrics.get("max_drawdown", 0.0)
        if dd < -0.10:
            triggered.append(self._make_alert(
                level="CRITICAL",
                code="HIGH_DRAWDOWN",
                msg=f"Max drawdown {dd*100:.1f}% melampaui batas 10%",
                value=dd,
            ))

        # Model drift
        if drift_result and drift_result.get("drifted"):
            triggered.append(self._make_alert(
                level="WARNING",
                code="MODEL_DRIFT",
                msg=(f"Model drift! Window acc={drift_result['window_accuracy']:.3f} "
                     f"vs baseline={drift_result['baseline_accuracy']:.3f}"),
                value=drift_result.get("delta"),
            ))

        # Kirim semua alert yang ter-trigger
        for alert in triggered:
            self._dispatch(alert)

        return triggered

    # ------------------------------------------------------------------
    # Dispatch ke channel
    # ------------------------------------------------------------------

    def _dispatch(self, alert: dict):
        """Kirim alert ke semua channel yang dikonfigurasi."""
        level_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(alert["level"], "📢")
        msg = f"{level_icon} [{alert['level']}] {alert['code']}: {alert['msg']}"

        # 1. Log
        if alert["level"] == "CRITICAL":
            logger.critical(msg)
        elif alert["level"] == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

        # 2. File log
        records = []
        if self.alert_log.exists():
            try:
                with open(self.alert_log) as f:
                    records = json.load(f)
            except Exception:
                pass
        records.append(alert)
        with open(self.alert_log, "w") as f:
            json.dump(records[-500:], f, indent=2, default=str)  # keep last 500

        # 3. Telegram (opsional — isi BOT_TOKEN dan CHAT_ID di config)
        tg_token = self.config.get("telegram", {}).get("bot_token")
        tg_chat  = self.config.get("telegram", {}).get("chat_id")
        if tg_token and tg_chat:
            self._send_telegram(tg_token, tg_chat, msg)

    def _send_telegram(self, token: str, chat_id: str, text: str):
        try:
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        except Exception as e:
            logger.debug(f"Telegram gagal: {e}")

    def _make_alert(self, level: str, code: str, msg: str, value=None) -> dict:
        return {
            "ts":    datetime.utcnow().isoformat(),
            "level": level,
            "code":  code,
            "msg":   msg,
            "value": value,
        }

    def load_recent_alerts(self, n: int = 50) -> list:
        if not self.alert_log.exists():
            return []
        try:
            with open(self.alert_log) as f:
                return json.load(f)[-n:]
        except Exception:
            return []


# ======================================================================
# RetrainScheduler — jadwal retrain otomatis
# ======================================================================

class RetrainScheduler:
    """
    Cek apakah model perlu diretrain berdasarkan:
    1. Interval waktu (default 7 hari)
    2. Deteksi drift
    3. Win rate jatuh di bawah threshold
    """

    def __init__(self, config: dict):
        self.config   = config
        self.interval = config["monitoring"]["retrain_interval_days"]
        self.log_dir  = Path(config["paths"]["logs_dir"])
        self.state_file = self.log_dir / "retrain_state.json"

    def should_retrain(self, metrics: dict = None, drift_result: dict = None) -> tuple:
        """
        Kembalikan (bool, reason_str).
        True berarti retrain harus dilakukan sekarang.
        """
        state = self._load_state()
        last_ts = state.get("last_retrain")

        reasons = []

        # 1. Cek interval waktu
        if last_ts:
            last_dt = datetime.fromisoformat(last_ts)
            days_since = (datetime.utcnow() - last_dt).days
            if days_since >= self.interval:
                reasons.append(f"sudah {days_since} hari sejak retrain terakhir")
        else:
            reasons.append("belum pernah retrain")

        # 2. Cek drift
        if drift_result and drift_result.get("drifted"):
            reasons.append(f"model drift (delta={drift_result.get('delta', '?')})")

        # 3. Cek win rate
        if metrics:
            wr = metrics.get("win_rate", 1.0)
            if wr < self.config["monitoring"]["min_win_rate"]:
                reasons.append(f"win rate {wr*100:.1f}% di bawah threshold")

        should = len(reasons) > 0
        return should, " | ".join(reasons) if reasons else "tidak perlu"

    def mark_retrained(self):
        """Tandai bahwa retrain baru saja selesai."""
        state = self._load_state()
        state["last_retrain"] = datetime.utcnow().isoformat()
        state["retrain_count"] = state.get("retrain_count", 0) + 1
        self._save_state(state)
        logger.info(f"✅ Retrain state diperbarui (total: {state['retrain_count']}x)")

    def get_status(self) -> dict:
        state = self._load_state()
        last_ts = state.get("last_retrain")
        days_since = None
        if last_ts:
            days_since = (datetime.utcnow() - datetime.fromisoformat(last_ts)).days
        return {
            "last_retrain":   last_ts,
            "days_since":     days_since,
            "retrain_count":  state.get("retrain_count", 0),
            "next_retrain_in": max(0, self.interval - (days_since or 0)),
        }

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self, state: dict):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
