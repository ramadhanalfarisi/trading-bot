"""
scripts/optimize_tp_sl.py

Grid search untuk menemukan kombinasi SL/TP (berbasis ATR multiplier)
yang memaksimalkan metrik backtest (default: total_pnl).

Usage:
    python scripts/optimize_tp_sl.py --symbol EURUSD --timeframe H1

Output:
    logs/optimize_tp_sl_results.csv
"""
import argparse
import itertools
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from pathlib import Path as _Path
# Ensure project root is on sys.path so `from data...` imports work when
# running this script directly (e.g. `python scripts/optimize_tp_sl.py`).
ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.collector import MT5Collector
from data.feature_engineering import FeatureEngineer
from data.preprocessor import Preprocessor
from models.lstm_model import LSTMTrainer
from models.xgboost_model import XGBoostTrainer
from models.ensemble import EnsemblePredictor
from models.backtest import Backtester


def load_data_and_models(cfg, symbol, timeframe):
    collector = MT5Collector(cfg)
    try:
        df_raw = collector.load_from_file(symbol, timeframe)
    except FileNotFoundError:
        logger.warning("File tidak ada, mencoba koneksi MT5 (jika tersedia).")
        collector.connect()
        df_raw = collector.get_ohlcv(symbol, timeframe, cfg["data"]["bars_history"]) 
        collector.disconnect()

    fe = FeatureEngineer(cfg)
    df_feat = fe.build_features(df_raw, add_labels=False)

    pre = Preprocessor(cfg)
    try:
        pre.load_scaler()
    except Exception:
        logger.warning("Scaler tidak ditemukan — coba lanjut tanpa scaling jika memungkinkan.")

    feat_cols = pre.feature_cols

    xgb = XGBoostTrainer(cfg)
    try:
        xgb.load_model()
    except Exception as e:
        logger.error(f"Gagal muat XGBoost model: {e}")
        raise

    lstm = LSTMTrainer(cfg)
    use_lstm = False
    try:
        lstm.build_model(len(feat_cols))
        lstm.load_model()
        use_lstm = True
    except Exception:
        logger.warning("LSTM tidak tersedia — fallback ke XGBoost untuk proba LSTM.")

    ensemble = EnsemblePredictor(cfg)
    ensemble.load()

    return collector, df_feat, pre, feat_cols, xgb, lstm, use_lstm, ensemble


def build_probabilities(df_feat, pre, feat_cols, xgb, lstm, use_lstm, cfg):
    X_all = pre.transform(df_feat[feat_cols])
    xgb_proba = xgb.predict_proba(X_all)

    if use_lstm:
        seq_len = cfg["lstm"]["sequence_length"]
        if len(X_all) >= seq_len:
            X_seq = np.array([X_all[i - seq_len:i] for i in range(seq_len, len(X_all))], dtype=np.float32)
            lstm_proba = lstm.predict_proba(X_seq)
            n = len(lstm_proba)
            xgb_proba_aligned = xgb_proba[-n:]
            df_bt = df_feat.iloc[-n:]
        else:
            logger.warning("Data terlalu sedikit untuk LSTM sequence — gunakan XGBoost proba saja.")
            lstm_proba = xgb_proba
            xgb_proba_aligned = xgb_proba
            df_bt = df_feat
    else:
        lstm_proba = xgb_proba
        xgb_proba_aligned = xgb_proba
        df_bt = df_feat

    return lstm_proba, xgb_proba_aligned, df_bt


def run_grid_search(cfg, df_bt, lstm_proba, xgb_proba_aligned, ensemble, sl_grid, tp_grid, symbol_point):
    results = []
    # ensure shapes
    if lstm_proba.ndim == 1:
        lstm_proba = lstm_proba[np.newaxis, :]
    if xgb_proba_aligned.ndim == 1:
        xgb_proba_aligned = xgb_proba_aligned[np.newaxis, :]

    for sl_mult, tp_mult in itertools.product(sl_grid, tp_grid):
        # temporary adjust risk multipliers
        ensemble.risk_cfg["sl_atr_multiplier"] = sl_mult
        ensemble.risk_cfg["tp_atr_multiplier"] = tp_mult

        results_list = ensemble.predict(lstm_proba, xgb_proba_aligned)
        if isinstance(results_list, dict):
            results_list = [results_list]
        signals = [r["signal"] for r in results_list]

        # backtest config must reflect multipliers
        # create a simple config wrapper for Backtester
        fake_cfg = {"risk": ensemble.risk_cfg, "data": {"point": symbol_point}, "backtest": {}}
        bt = Backtester(fake_cfg)
        metrics = bt.run(df_bt.iloc[:len(signals)], signals, verbose=False)

        results.append({
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            **metrics,
        })

    return results


def save_results(results, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results[0].keys()) if results else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Optimize TP/SL multipliers via grid search")
    p.add_argument("--symbol", type=str, default="EURUSD")
    p.add_argument("--timeframe", type=str, default="H1")
    p.add_argument("--sl-min", type=float, default=0.5)
    p.add_argument("--sl-max", type=float, default=3.0)
    p.add_argument("--sl-steps", type=int, default=6)
    p.add_argument("--tp-min", type=float, default=1.0)
    p.add_argument("--tp-max", type=float, default=6.0)
    p.add_argument("--tp-steps", type=int, default=10)
    p.add_argument("--config", type=str, default="config/config.yaml")
    args = p.parse_args()

    # load config
    import yaml
    cfg = yaml.safe_load(open(args.config))

    collector, df_feat, pre, feat_cols, xgb, lstm, use_lstm, ensemble = load_data_and_models(cfg, args.symbol, args.timeframe)
    lstm_proba, xgb_proba_aligned, df_bt = build_probabilities(df_feat, pre, feat_cols, xgb, lstm, use_lstm, cfg)

    sl_grid = np.linspace(args.sl_min, args.sl_max, args.sl_steps)
    tp_grid = np.linspace(args.tp_min, args.tp_max, args.tp_steps)

    symbol_point = cfg.get("data", {}).get("point", 0.00001)

    results = run_grid_search(cfg, df_bt, lstm_proba, xgb_proba_aligned, ensemble, sl_grid, tp_grid, symbol_point)

    out_file = Path(cfg.get("paths", {}).get("logs_dir", "logs")) / f"optimize_tp_sl_{args.symbol}_{args.timeframe}.csv"
    if results:
        save_results(results, out_file)
        logger.info(f"Hasil disimpan di {out_file}")
    else:
        logger.warning("Tidak ada hasil (mungkin error saat eksekusi).")


if __name__ == "__main__":
    main()
