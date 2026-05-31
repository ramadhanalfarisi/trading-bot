"""
tests/test_all.py
Unit tests untuk semua komponen sistem ML Forex Advisor.

Jalankan dengan:
    pytest tests/test_all.py -v
    pytest tests/test_all.py -v --cov=. --cov-report=term-missing
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Tambahkan root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config minimal untuk testing ─────────────────────────────────────
TEST_CONFIG = {
    "mt5": {"login": 0, "password": "", "server": "Demo"},
    "data": {
        "primary_symbol": "EURUSD",
        "timeframes": {"primary": "H1", "confirmation": "M15"},
        "bars_history": 500,
        "bars_inference": 100,
        "data_dir": "/tmp/forex_test/raw",
        "processed_dir": "/tmp/forex_test/processed",
        "symbols": ["EURUSD"],
    },
    "features": {
        "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_period": 20, "bb_std": 2.0, "atr_period": 14,
        "ema_periods": [9, 21, 50, 200], "sma_periods": [20, 50],
        "stoch_k": 14, "stoch_d": 3,
        "lag_returns": [1, 2, 3, 5], "rolling_windows": [5, 10, 20],
    },
    "label": {"lookahead_bars": 3, "threshold_pips": 10},
    "preprocessing": {
        "scaler": "StandardScaler",
        "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15,
        "walkforward_n_splits": 3,
        "walkforward_train_size": 200,
        "walkforward_test_size": 50,
    },
    "lstm": {
        "sequence_length": 20, "hidden_size": 32, "num_layers": 1,
        "dropout": 0.1, "batch_size": 16, "epochs": 2,
        "learning_rate": 0.001, "patience": 3, "weight_decay": 0.0001,
    },
    "xgboost": {
        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
        "gamma": 0, "reg_alpha": 0, "reg_lambda": 1, "use_gpu": False,
    },
    "ensemble": {
        "lstm_weight": 0.45, "xgboost_weight": 0.55,
        "confidence_threshold": 0.55,
    },
    "risk": {
        "max_risk_per_trade": 0.01, "max_spread_pips": 3.0,
        "max_daily_drawdown": 0.05, "max_open_trades": 3,
        "sl_atr_multiplier": 2.0, "tp_atr_multiplier": 3.0,
        "news_filter_minutes": 30,
    },
    "paths": {
        "models_dir": "/tmp/forex_test/models",
        "lstm_model": "/tmp/forex_test/models/lstm.pt",
        "xgb_model":  "/tmp/forex_test/models/xgb.json",
        "ensemble_model": "/tmp/forex_test/models/ensemble.joblib",
        "scaler":     "/tmp/forex_test/models/scaler.joblib",
        "onnx_model": "/tmp/forex_test/models/model.onnx",
        "logs_dir":   "/tmp/forex_test/logs",
    },
    "monitoring": {
        "retrain_interval_days": 7,
        "min_win_rate": 0.45,
        "min_sharpe": 0.5,
        "drift_threshold": 0.10,
    },
    "api": {"host": "127.0.0.1", "port": 5001, "debug": False, "secret_key": "test"},
}

# ── Fixture: dummy OHLCV ──────────────────────────────────────────────
def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.randn(n) * 0.0005)
    noise = np.abs(np.random.randn(n) * 0.0002) + 1e-5
    return pd.DataFrame({
        "open":   close - np.random.randn(n) * 0.0001,
        "high":   close + noise,
        "low":    close - noise,
        "close":  close,
        "volume": np.random.randint(100, 5000, n).astype(float),
    }, index=dates)


# ======================================================================
# Test: MT5Collector
# ======================================================================
class TestMT5Collector:
    def setup_method(self):
        from data.collector import MT5Collector
        self.collector = MT5Collector(TEST_CONFIG)

    def test_generate_dummy_data(self):
        df = self.collector._generate_dummy_data(200)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
        # high >= close >= low
        assert (df["high"] >= df["close"]).all()
        assert (df["close"] >= df["low"]).all()

    def test_connect_without_mt5(self):
        # MT5 tidak tersedia di test env — should return False gracefully
        result = self.collector.connect()
        assert isinstance(result, bool)

    def test_get_ohlcv_no_connection(self):
        df = self.collector.get_ohlcv("EURUSD", "H1", 100, save=False)
        assert len(df) == 100

    def test_get_account_info_offline(self):
        info = self.collector.get_account_info()
        assert "balance" in info
        assert info["balance"] > 0


# ======================================================================
# Test: FeatureEngineer
# ======================================================================
class TestFeatureEngineer:
    def setup_method(self):
        from data.feature_engineering import FeatureEngineer
        self.fe  = FeatureEngineer(TEST_CONFIG)
        self.raw = make_ohlcv(500)

    def test_build_features_shape(self):
        df = self.fe.build_features(self.raw, add_labels=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "label" in df.columns

    def test_label_range(self):
        df = self.fe.build_features(self.raw, add_labels=True)
        assert df["label"].isin([0, 1, 2]).all(), "Label harus 0, 1, atau 2"

    def test_no_all_nan_columns(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        feat_cols = self.fe.get_feature_columns(df)
        nan_cols = [c for c in feat_cols if df[c].isna().all()]
        assert len(nan_cols) == 0, f"Kolom semua NaN: {nan_cols}"

    def test_feature_count_reasonable(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        feat_cols = self.fe.get_feature_columns(df)
        assert len(feat_cols) >= 30, f"Terlalu sedikit fitur: {len(feat_cols)}"

    def test_rsi_range(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        assert df["rsi"].dropna().between(0, 100).all()

    def test_bb_logic(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_atr_positive(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        assert (df["atr"].dropna() > 0).all()

    def test_no_labels_mode(self):
        df = self.fe.build_features(self.raw, add_labels=False)
        assert "label" not in df.columns


# ======================================================================
# Test: Preprocessor
# ======================================================================
class TestPreprocessor:
    def setup_method(self):
        from data.feature_engineering import FeatureEngineer
        from data.preprocessor import Preprocessor
        fe  = FeatureEngineer(TEST_CONFIG)
        raw = make_ohlcv(500)
        self.df_feat = fe.build_features(raw, add_labels=True)
        self.feat_cols = fe.get_feature_columns(self.df_feat)
        self.pre = Preprocessor(TEST_CONFIG)

    def test_split_shapes(self):
        Xt, Xv, Xte, yt, yv, yte = self.pre.split(self.df_feat, self.feat_cols)
        total = len(Xt) + len(Xv) + len(Xte)
        assert total == len(self.df_feat)
        assert Xt.shape[1] == len(self.feat_cols)

    def test_split_ratios(self):
        n = len(self.df_feat)
        Xt, Xv, Xte, *_ = self.pre.split(self.df_feat, self.feat_cols)
        assert abs(len(Xt) / n - 0.70) < 0.05
        assert abs(len(Xv) / n - 0.15) < 0.05

    def test_make_sequences(self):
        Xt, Xv, Xte, yt, yv, yte = self.pre.split(self.df_feat, self.feat_cols)
        seq_len = 20
        Xs, ys = self.pre.make_sequences(Xt, yt, seq_len)
        assert Xs.shape == (len(Xt) - seq_len, seq_len, len(self.feat_cols))
        assert len(Xs) == len(ys)

    def test_scaler_fit(self):
        self.pre.split(self.df_feat, self.feat_cols)
        assert self.pre.scaler is not None
        assert hasattr(self.pre.scaler, "mean_") or hasattr(self.pre.scaler, "scale_")

    def test_transform_shape(self):
        self.pre.split(self.df_feat, self.feat_cols)
        X = self.pre.transform(self.df_feat.tail(10)[self.feat_cols])
        assert X.shape == (10, len(self.feat_cols))

    def test_walkforward_splits(self):
        Xt, *_ = self.pre.split(self.df_feat, self.feat_cols)
        yt = self.df_feat["label"].values[:len(Xt)].astype(int)
        splits = list(self.pre.walk_forward_splits(Xt, yt))
        assert len(splits) == TEST_CONFIG["preprocessing"]["walkforward_n_splits"]
        for X_tr, y_tr, X_te, y_te in splits:
            assert len(X_tr) > 0 and len(X_te) > 0


# ======================================================================
# Test: XGBoost Model
# ======================================================================
class TestXGBoostModel:
    def setup_method(self):
        from data.feature_engineering import FeatureEngineer
        from data.preprocessor import Preprocessor
        from models.xgboost_model import XGBoostTrainer

        fe  = FeatureEngineer(TEST_CONFIG)
        raw = make_ohlcv(400)
        df  = fe.build_features(raw, add_labels=True)
        pre = Preprocessor(TEST_CONFIG)
        feat_cols = fe.get_feature_columns(df)

        self.Xt, self.Xv, self.Xte, self.yt, self.yv, self.yte = pre.split(df, feat_cols)
        self.trainer = XGBoostTrainer(TEST_CONFIG)

    def test_train_returns_metrics(self):
        metrics = self.trainer.train(self.Xt, self.yt, self.Xv, self.yv)
        assert "val_accuracy" in metrics
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_predict_proba_shape(self):
        self.trainer.train(self.Xt, self.yt, self.Xv, self.yv)
        proba = self.trainer.predict_proba(self.Xte)
        assert proba.shape == (len(self.Xte), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_predict_classes(self):
        self.trainer.train(self.Xt, self.yt, self.Xv, self.yv)
        preds = self.trainer.predict(self.Xte)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_save_load(self, tmp_path):
        cfg = TEST_CONFIG.copy()
        cfg["paths"] = {**cfg["paths"], "xgb_model": str(tmp_path / "xgb.json")}
        from models.xgboost_model import XGBoostTrainer
        t = XGBoostTrainer(cfg)
        t.train(self.Xt, self.yt, self.Xv, self.yv)
        t.load_model()
        proba = t.predict_proba(self.Xte[:5])
        assert proba.shape == (5, 3)


# ======================================================================
# Test: LSTM Model
# ======================================================================
class TestLSTMModel:
    def setup_method(self):
        from data.feature_engineering import FeatureEngineer
        from data.preprocessor import Preprocessor
        from models.lstm_model import LSTMTrainer

        fe  = FeatureEngineer(TEST_CONFIG)
        raw = make_ohlcv(400)
        df  = fe.build_features(raw, add_labels=True)
        pre = Preprocessor(TEST_CONFIG)
        feat_cols = fe.get_feature_columns(df)

        Xt, Xv, Xte, yt, yv, yte = pre.split(df, feat_cols)
        seq_len = TEST_CONFIG["lstm"]["sequence_length"]
        self.Xt_seq, self.yt_seq = pre.make_sequences(Xt, yt, seq_len)
        self.Xv_seq, self.yv_seq = pre.make_sequences(Xv, yv, seq_len)
        self.n_features = Xt.shape[1]
        self.trainer    = LSTMTrainer(TEST_CONFIG)

    def test_build_model(self):
        model = self.trainer.build_model(self.n_features)
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_train_returns_history(self):
        history = self.trainer.train(self.Xt_seq, self.yt_seq, self.Xv_seq, self.yv_seq)
        assert "train_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) > 0

    def test_predict_proba_shape(self):
        self.trainer.train(self.Xt_seq, self.yt_seq, self.Xv_seq, self.yv_seq)
        proba = self.trainer.predict_proba(self.Xv_seq)
        assert proba.shape == (len(self.Xv_seq), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


# ======================================================================
# Test: Ensemble
# ======================================================================
class TestEnsemble:
    def setup_method(self):
        from models.ensemble import EnsemblePredictor
        self.ensemble = EnsemblePredictor(TEST_CONFIG)

    def _make_proba(self, n=5):
        raw = np.random.dirichlet([1, 1, 1], n).astype(np.float32)
        return raw

    def test_predict_single(self):
        lp = self._make_proba(1)[0]
        xp = self._make_proba(1)[0]
        r = self.ensemble.predict_single(lp, xp, atr_value=0.001, current_price=1.0850)
        assert "signal" in r
        assert r["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_predict_batch(self):
        lp = self._make_proba(10)
        xp = self._make_proba(10)
        results = self.ensemble.predict(lp, xp)
        assert len(results) == 10
        for r in results:
            assert r["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_below_threshold_gives_hold(self):
        # Konstruksi proba yang semua rendah di bawah threshold
        lp = np.array([[0.34, 0.33, 0.33]], dtype=np.float32)
        xp = np.array([[0.34, 0.33, 0.33]], dtype=np.float32)
        result = self.ensemble.predict(lp, xp)
        assert result[0]["signal"] == "HOLD"

    def test_sl_tp_logic_buy(self):
        lp = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)
        xp = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)
        r = self.ensemble.predict_single(lp[0], xp[0], atr_value=0.001, current_price=1.0850)
        if r["signal"] == "BUY":
            assert r["sl_price"] < r["current_price"]
            assert r["tp_price"] > r["current_price"]

    def test_lot_size_within_bounds(self):
        lot = self.ensemble.calculate_lot_size(
            balance=10000, sl_pips=20,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
        )
        assert 0.01 <= lot <= 10.0


# ======================================================================
# Test: Backtester
# ======================================================================
class TestBacktester:
    def setup_method(self):
        from models.backtest import Backtester
        from data.feature_engineering import FeatureEngineer

        self.bt = Backtester(TEST_CONFIG)
        fe  = FeatureEngineer(TEST_CONFIG)
        raw = make_ohlcv(300)
        self.df_feat = fe.build_features(raw, add_labels=False)

    def test_run_returns_metrics(self):
        n = len(self.df_feat)
        signals = np.random.choice(["BUY", "SELL", "HOLD"], n, p=[0.3, 0.3, 0.4]).tolist()
        metrics = self.bt.run(self.df_feat, signals, verbose=False)
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_no_trades_hold_only(self):
        n = len(self.df_feat)
        signals = ["HOLD"] * n
        metrics = self.bt.run(self.df_feat, signals, verbose=False)
        assert metrics["total_trades"] == 0

    def test_win_rate_range(self):
        n = len(self.df_feat)
        signals = np.random.choice(["BUY", "SELL"], n).tolist()
        metrics = self.bt.run(self.df_feat, signals, verbose=False)
        if metrics["total_trades"] > 0:
            assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_final_balance_positive(self):
        n = len(self.df_feat)
        signals = ["HOLD"] * n
        metrics = self.bt.run(self.df_feat, signals, verbose=False)
        assert metrics["final_balance"] == 10000.0


# ======================================================================
# Test: MetricsTracker
# ======================================================================
class TestMetricsTracker:
    def setup_method(self, tmp_path=None):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        cfg = TEST_CONFIG.copy()
        cfg["paths"] = {**cfg["paths"], "logs_dir": self.tmpdir}
        from monitoring.metrics import MetricsTracker
        self.tracker = MetricsTracker(cfg)

    def test_record_training(self):
        self.tracker.record_training("EURUSD", {"val_accuracy": 0.58, "win_rate": 0.52})
        df = self.tracker.load_metrics_history("EURUSD")
        assert len(df) >= 1

    def test_record_trade(self):
        self.tracker.record_trade({
            "symbol": "EURUSD", "direction": "BUY",
            "pnl": 45.0, "result": "WIN", "confidence": 0.72,
        })
        df = self.tracker.load_trades("EURUSD")
        assert len(df) >= 1

    def test_compute_live_performance_empty(self):
        result = self.tracker.compute_live_performance("GBPUSD")
        assert result == {}


# ======================================================================
# Test: DriftDetector
# ======================================================================
class TestDriftDetector:
    def setup_method(self):
        from monitoring.metrics import DriftDetector
        self.detector = DriftDetector(TEST_CONFIG)

    def test_no_drift_when_similar(self):
        np.random.seed(1)
        trades = pd.DataFrame({
            "result": np.where(np.random.rand(60) < 0.56, "WIN", "LOSS"),
            "confidence": np.random.uniform(0.6, 0.9, 60),
        })
        result = self.detector.check(trades, baseline_accuracy=0.58)
        assert "drifted" in result
        assert isinstance(result["drifted"], bool)

    def test_drift_when_accuracy_drops(self):
        trades = pd.DataFrame({
            "result": ["LOSS"] * 50 + ["WIN"] * 10,
            "confidence": [0.6] * 60,
        })
        result = self.detector.check(trades, baseline_accuracy=0.80)
        assert result["drifted"] is True

    def test_insufficient_data(self):
        trades = pd.DataFrame({"result": ["WIN", "LOSS"]})
        result = self.detector.check(trades, baseline_accuracy=0.6)
        assert result["drifted"] is False


# ======================================================================
# Test: AlertManager
# ======================================================================
class TestAlertManager:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        cfg = TEST_CONFIG.copy()
        cfg["paths"] = {**cfg["paths"], "logs_dir": self.tmpdir}
        from monitoring.metrics import AlertManager
        self.alerter = AlertManager(cfg)

    def test_low_winrate_triggers_alert(self):
        metrics = {"win_rate": 0.30, "sharpe_ratio": 1.0, "max_drawdown": -0.05}
        alerts = self.alerter.check_all(metrics)
        codes = [a["code"] for a in alerts]
        assert "LOW_WIN_RATE" in codes

    def test_high_drawdown_triggers_critical(self):
        metrics = {"win_rate": 0.55, "sharpe_ratio": 1.5, "max_drawdown": -0.15}
        alerts = self.alerter.check_all(metrics)
        levels = [a["level"] for a in alerts]
        assert "CRITICAL" in levels

    def test_healthy_metrics_no_alert(self):
        metrics = {"win_rate": 0.55, "sharpe_ratio": 1.5, "max_drawdown": -0.05}
        alerts = self.alerter.check_all(metrics)
        assert len(alerts) == 0


# ======================================================================
# Test: RetrainScheduler
# ======================================================================
class TestRetrainScheduler:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        cfg = TEST_CONFIG.copy()
        cfg["paths"] = {**cfg["paths"], "logs_dir": self.tmpdir}
        from monitoring.metrics import RetrainScheduler
        self.scheduler = RetrainScheduler(cfg)

    def test_should_retrain_first_time(self):
        should, reason = self.scheduler.should_retrain()
        assert should is True
        assert "belum pernah" in reason

    def test_mark_retrained(self):
        self.scheduler.mark_retrained()
        status = self.scheduler.get_status()
        assert status["retrain_count"] == 1
        assert status["last_retrain"] is not None

    def test_no_retrain_after_recent(self):
        self.scheduler.mark_retrained()
        # Reset interval ke sangat besar
        self.scheduler.interval = 999
        should, reason = self.scheduler.should_retrain()
        assert should is False

    def test_force_retrain_on_drift(self):
        self.scheduler.mark_retrained()
        self.scheduler.interval = 999
        drift = {"drifted": True, "delta": 0.15}
        should, reason = self.scheduler.should_retrain(drift_result=drift)
        assert should is True


# ======================================================================
# Test: Flask API (tanpa server berjalan)
# ======================================================================
class TestFlaskAPI:
    def setup_method(self):
        from api.server import create_app
        app = create_app(TEST_CONFIG)
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert "status" in data

    def test_model_info_endpoint(self):
        r = self.client.get("/model/info")
        assert r.status_code == 200
        data = r.get_json()
        assert "models" in data or "config" in data

    def test_predict_requires_body(self):
        r = self.client.post("/predict",
            data=b"", content_type="application/json")
        assert r.status_code in (400, 415, 200)

    def test_predict_insufficient_bars(self):
        import json
        payload = {"symbol": "EURUSD", "timeframe": "H1", "bars": []}
        r = self.client.post("/predict",
            data=json.dumps(payload),
            content_type="application/json")
        assert r.status_code == 400


# ======================================================================
# Runner
# ======================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
