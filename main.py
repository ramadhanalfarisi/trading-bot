"""
main.py
Entry point utama sistem ML Forex Advisor.

Penggunaan:
    python main.py --mode collect  --symbol EURUSD --timeframe H1 --bars 5000
    python main.py --mode train    --symbol EURUSD [--skip-lstm] [--from-file]
    python main.py --mode backtest --symbol EURUSD
    python main.py --mode predict  --symbol EURUSD
    python main.py --mode api
    python main.py --mode monitor
    python main.py --mode scheduler        # loop retrain otomatis
"""
import argparse
import sys
import yaml
from pathlib import Path
from loguru import logger


# ── Logger setup ──────────────────────────────────────────────────────
def setup_logger(log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(f"{log_dir}/forex_ml_{{time:YYYY-MM-DD}}.log",
               rotation="1 day", retention="14 days", level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}")


# ── Load config ───────────────────────────────────────────────────────
def load_config(path: str = "config/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.error(f"Config tidak ditemukan: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


# ======================================================================
# MODE: collect
# ======================================================================
def mode_collect(cfg: dict, args):
    from data.collector import MT5Collector
    symbol    = args.symbol    or cfg["data"]["primary_symbol"]
    timeframe = args.timeframe or cfg["data"]["timeframes"]["primary"]
    bars      = args.bars      or cfg["data"]["bars_history"]

    logger.info(f"📥 Collecting data | {symbol} {timeframe} | {bars} bar")
    collector = MT5Collector(cfg)

    connected = collector.connect()
    if not connected:
        logger.warning("MT5 tidak terhubung — menggunakan data simulasi untuk demo.")

    df = collector.get_ohlcv(symbol, timeframe, bars, save=True)
    logger.info(f"✅ Data tersimpan: {len(df)} bar | {df.index[0]} → {df.index[-1]}")
    collector.disconnect()


# ======================================================================
# MODE: train
# ======================================================================
def mode_train(cfg: dict, args):
    from models.trainer import TrainingPipeline
    from monitoring.metrics import MetricsTracker, RetrainScheduler

    symbol     = args.symbol or cfg["data"]["primary_symbol"]
    from_file  = args.from_file
    skip_lstm  = args.skip_lstm

    logger.info(f"🚀 Training | Symbol: {symbol} | from_file: {from_file} | skip_lstm: {skip_lstm}")

    pipeline = TrainingPipeline(cfg)
    results  = pipeline.run(symbol=symbol, from_file=from_file, skip_lstm=skip_lstm)

    # Catat metrik ke tracker
    tracker   = MetricsTracker(cfg)
    scheduler = RetrainScheduler(cfg)
    tracker.record_training(symbol, results.get("backtest", {}))
    scheduler.mark_retrained()

    logger.info("✅ Training selesai.")
    return results


# ======================================================================
# MODE: backtest
# ======================================================================
def mode_backtest(cfg: dict, args):
    import pandas as pd
    from data.collector import MT5Collector
    from data.feature_engineering import FeatureEngineer
    from data.preprocessor import Preprocessor
    from models.lstm_model import LSTMTrainer
    from models.xgboost_model import XGBoostTrainer
    from models.ensemble import EnsemblePredictor
    from models.backtest import Backtester
    import numpy as np

    symbol    = args.symbol    or cfg["data"]["primary_symbol"]
    timeframe = args.timeframe or cfg["data"]["timeframes"]["primary"]

    logger.info(f"📊 Backtest | {symbol} {timeframe}")

    # Load data
    collector = MT5Collector(cfg)
    try:
        df_raw = collector.load_from_file(symbol, timeframe)
    except FileNotFoundError:
        logger.warning("File tidak ada, menggunakan data simulasi.")
        collector.connect()
        df_raw = collector.get_ohlcv(symbol, timeframe, cfg["data"]["bars_history"])
        collector.disconnect()

    # Build features using full loaded data (allow optional date filtering)
    # Backtest should operate on the requested historical range, not the
    # small inference window used for live/predict.
    # Optionally filter by --start-date / --end-date (YYYY-MM-DD)
    try:
        if getattr(args, "start_date", None) or getattr(args, "end_date", None):
            start = pd.to_datetime(args.start_date) if args.start_date else df_raw.index.min()
            end = pd.to_datetime(args.end_date) if args.end_date else df_raw.index.max()
            # make start/end timezone-aware to match df_raw.index if needed
            idx_tz = getattr(df_raw.index, 'tz', None)
            if idx_tz is not None:
                if start.tzinfo is None:
                    start = start.tz_localize(idx_tz)
                if end.tzinfo is None:
                    end = end.tz_localize(idx_tz)

            df_raw = df_raw[(df_raw.index >= start) & (df_raw.index <= end)]
            if df_raw.empty:
                logger.error("Tidak ada data untuk range tanggal yang diminta.")
                sys.exit(1)
            # warn if requested range not fully covered
            if df_raw.index.min() > start or df_raw.index.max() < end:
                logger.warning(
                    f"Requested range {start} -> {end} not fully available. "
                    f"Available: {df_raw.index.min()} -> {df_raw.index.max()}"
                )
            logger.info(f"Filtered data to range: {start} -> {end} | {len(df_raw)} bars")
    except Exception as e:
        logger.warning(f"Gagal memfilter range tanggal: {e}")

    fe = FeatureEngineer(cfg)
    df_feat = fe.build_features(df_raw, add_labels=False)

    # Load preprocessor & models
    pre = Preprocessor(cfg)
    pre.load_scaler()
    feat_cols = pre.feature_cols

    xgb = XGBoostTrainer(cfg)
    xgb.load_model()

    ensemble = EnsemblePredictor(cfg)
    ensemble.load()

    X_all = pre.transform(df_feat[feat_cols])
    xgb_proba = xgb.predict_proba(X_all)

    seq_len = cfg["lstm"]["sequence_length"]
    lstm = LSTMTrainer(cfg)
    try:
        lstm.build_model(len(feat_cols))
        lstm.load_model()
        Xs = []
        for i in range(seq_len, len(X_all)):
            Xs.append(X_all[i - seq_len:i])
        X_seq = np.array(Xs, dtype=np.float32)
        lstm_proba = lstm.predict_proba(X_seq)
        n = len(lstm_proba)
        xgb_proba_aligned = xgb_proba[-n:]
        df_bt = df_feat.iloc[-n:]
    except Exception as e:
        logger.warning(f"LSTM tidak tersedia ({e}), pakai XGBoost saja.")
        lstm_proba = xgb_proba
        n = len(xgb_proba)
        xgb_proba_aligned = xgb_proba
        df_bt = df_feat

    results_list = ensemble.predict(lstm_proba, xgb_proba_aligned)
    if isinstance(results_list, dict):
        results_list = [results_list]
    signals = [r["signal"] for r in results_list]

    bt = Backtester(cfg)
    metrics = bt.run(df_bt.iloc[:len(signals)], signals)

    # Save and print trade history summary
    try:
        trades = getattr(bt, "last_trades", []) or []
        if trades:
            df_trades = pd.DataFrame(trades)
            from datetime import datetime
            logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)
            out_path = logs_dir / f"trade_history_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(out_path, index=False)
            logger.info(f"Saved trade history to {out_path}")
            logger.info("Recent trades:")
            for t in trades[-10:]:
                logger.info(f"  {t.get('entry_time')} | {t.get('direction')} | entry={t.get('entry_price')} exit={t.get('exit_price')} pnl={t.get('pnl')} lot={t.get('lot')}")
        else:
            logger.info("No trades executed in this backtest.")
    except Exception as e:
        logger.warning(f"Gagal menyimpan ringkasan trade: {e}")

    logger.info("✅ Backtest selesai.")
    return metrics


# ======================================================================
# MODE: predict (satu prediksi)
# ======================================================================
def mode_ticktest(cfg: dict, args):
    import pandas as pd
    from data.collector import MT5Collector
    from api.predictor import ForexPredictor
    from models.tick_backtest import TickBacktester

    symbol = args.symbol or cfg["data"]["primary_symbol"]
    timeframe = args.tick_timeframe or cfg["data"]["timeframes"]["primary"]
    tick_file = args.tick_file
    tick_count = args.tick_count
    start_date = args.start_date or cfg["data"].get("tick_backtest_start_date")
    end_date = args.end_date or cfg["data"].get("tick_backtest_end_date")
    
    min_bars = estimate_required_bars(cfg)
    recommended_ticks = estimate_required_ticks(timeframe, min_bars)

    logger.info(f"🧪 TickTest | {symbol} | timeframe={timeframe}")
    
    if start_date and end_date:
        logger.info(f"   Date range: {start_date} to {end_date}")

    collector = MT5Collector(cfg)
    
    connected = collector.connect()
    if not connected:
        logger.warning("MT5 tidak terhubung — menggunakan data simulasi untuk demo.")
    # Load tick data - prefer date range over tick_file/tick_count
    if start_date and end_date:
        try:
            df_ticks = collector.collect_ticks_range(symbol, start_date, end_date, save=True)
        except FileNotFoundError:
            logger.warning(f"Tidak bisa mengunduh tick range {start_date}-{end_date}")
            # Fallback: try loading from file with filter
            try:
                df_ticks = collector.load_tick_history_filtered(
                    symbol, 
                    filename=tick_file,
                    start_date=start_date,
                    end_date=end_date
                )
            except FileNotFoundError:
                logger.error(f"Tick history file tidak ditemukan untuk {symbol}")
                sys.exit(1)
    else:
        # Original behavior: use tick_file or generate
        if tick_file is None:
            if tick_count is None:
                tick_count = recommended_ticks
                logger.info(f"🔧 Tick count default ditetapkan ke {tick_count} untuk {timeframe} agar tersedia minimal {min_bars} bar.")
            elif tick_count < recommended_ticks:
                logger.warning(
                    f"tick_count {tick_count} terlalu kecil untuk timeframe {timeframe}. "
                    f"Menaikkan ke {recommended_ticks} agar ada cukup bar ML."
                )
                tick_count = recommended_ticks
        else:
            logger.info(f"🔧 Menggunakan tick file {tick_file} dan tidak mengubah jumlah tick yang di-generate.")

        collector.connect()
        df_ticks = collector.collect_ticks(symbol, tick_count, save=True)
        collector.disconnect()

    predictor = ForexPredictor(cfg)
    use_model = True
    try:
        predictor.load_models()
    except Exception as e:
        logger.warning(f"Gagal memuat model ML: {e}. Fallback ke prediksi momentum sederhana.")
        use_model = False

    def predict_fn(df_bars: pd.DataFrame) -> dict:
        if use_model:
            try:
                return predictor.predict_from_ohlcv(df_bars)
            except Exception as e:
                logger.warning(f"Predictor gagal saat ticktest: {e}")
        if len(df_bars) < 2:
            return {"signal": "HOLD", "confidence": 0.0}
        last = df_bars["close"].iloc[-1]
        prev = df_bars["close"].iloc[-2]
        if last > prev:
            return {"signal": "BUY", "confidence": 0.60}
        if last < prev:
            return {"signal": "SELL", "confidence": 0.60}
        return {"signal": "HOLD", "confidence": 0.0}

    tick_bt = TickBacktester(cfg)
    result = tick_bt.run(df_ticks, symbol, predict_fn, timeframe=timeframe)

    metrics = result.get("metrics", {})
    logger.info("\n📊 TickTest Result:")
    logger.info(f"{'='*60}")
    for k, v in metrics.items():
        if k in ["leverage", "margin_requirement_pct", "initial_balance"]:
            logger.info(f"  {k:<25}: {v}")
    logger.info(f"{'-'*60}")
    for k, v in metrics.items():
        if k not in ["leverage", "margin_requirement_pct", "initial_balance"]:
            logger.info(f"  {k:<25}: {v}")
    logger.info(f"{'='*60}")

    trades = result.get("trades", [])
    if trades:
        logger.info(f"  Trade log: {len(trades)} trades")
    else:
        logger.info("  Tidak ada trade terbuka selama tick replay.")

    return result


def estimate_required_bars(cfg: dict) -> int:
    feature_cfg = cfg["features"]
    label_cfg = cfg["label"]
    return max(
        max(feature_cfg.get("ema_periods", [0])),
        max(feature_cfg.get("sma_periods", [0])),
        feature_cfg.get("bb_period", 0),
        feature_cfg.get("atr_period", 0),
        max(feature_cfg.get("lag_returns", [0])) + 1,
        label_cfg.get("lookahead_bars", 0) + 1,
    )


def estimate_required_ticks(timeframe: str, min_bars: int) -> int:
    seconds_per_bar = {
        "M1": 60,
        "M5": 5 * 60,
        "M15": 15 * 60,
        "M30": 30 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
    }.get(timeframe.upper(), 60)
    # Dummy tick generator uses 1-second spacing, jadi estimasi tick sama dengan detik.
    return max(min_bars * seconds_per_bar, 5000)


def mode_predict(cfg: dict, args):
    from api.predictor import ForexPredictor
    symbol    = args.symbol    or cfg["data"]["primary_symbol"]
    timeframe = args.timeframe or cfg["data"]["timeframes"]["primary"]

    logger.info(f"🔮 Prediksi | {symbol} {timeframe}")
    predictor = ForexPredictor(cfg)
    predictor.load_models()

    result = predictor.predict_live(symbol, timeframe)
    logger.info(f"\n{'='*50}")
    logger.info(f"  Signal    : {result['signal']}")
    logger.info(f"  Confidence: {result['confidence']*100:.1f}%")
    logger.info(f"  Harga     : {result.get('current_price', 'N/A')}")
    logger.info(f"  SL        : {result.get('sl_price', 'N/A')} ({result.get('sl_pips', 0)} pips)")
    logger.info(f"  TP        : {result.get('tp_price', 'N/A')} ({result.get('tp_pips', 0)} pips)")
    logger.info(f"  Lot       : {result.get('lot_size', 0)}")
    logger.info(f"  Risk OK   : {result.get('risk_passed', False)}")
    logger.info(f"{'='*50}")
    return result


# ======================================================================
# MODE: api
# ======================================================================
def mode_api(cfg: dict, args):
    from api.server import run_server
    run_server(cfg)


# ======================================================================
# MODE: monitor (tampilkan status sekali)
# ======================================================================
def mode_monitor(cfg: dict, args):
    from monitoring.metrics import MetricsTracker, DriftDetector, AlertManager, RetrainScheduler

    symbol = args.symbol or cfg["data"]["primary_symbol"]
    tracker   = MetricsTracker(cfg)
    detector  = DriftDetector(cfg)
    alerter   = AlertManager(cfg)
    scheduler = RetrainScheduler(cfg)

    logger.info(f"\n{'='*55}")
    logger.info(f"  MONITORING REPORT — {symbol}")
    logger.info(f"{'='*55}")

    # Live performance
    live_metrics = tracker.compute_live_performance(symbol)
    if live_metrics:
        logger.info(f"\n📊 Live Performance:")
        for k, v in live_metrics.items():
            logger.info(f"   {k:<20}: {v}")
    else:
        logger.info("  Belum ada data trade. Jalankan EA dan trade dulu.")

    # Drift check
    trades_df = tracker.load_trades(symbol)
    if not trades_df.empty and "result" in trades_df.columns:
        drift = detector.check(trades_df, baseline_accuracy=0.58)
        logger.info(f"\n🔬 Drift Detection: {'⚠️ DRIFT' if drift['drifted'] else '✅ OK'}")
        logger.info(f"   Window acc: {drift.get('window_accuracy')} | Delta: {drift.get('delta')}")
    else:
        drift = None

    # Alert check
    if live_metrics:
        alerts = alerter.check_all(live_metrics, drift)
        if alerts:
            logger.warning(f"\n🚨 {len(alerts)} Alert aktif!")
        else:
            logger.info("\n✅ Tidak ada alert aktif.")

    # Retrain schedule
    sched_status = scheduler.get_status()
    should, reason = scheduler.should_retrain(live_metrics, drift)
    logger.info(f"\n🔄 Retrain Scheduler:")
    logger.info(f"   Last retrain  : {sched_status.get('last_retrain', 'belum pernah')}")
    logger.info(f"   Days since    : {sched_status.get('days_since', 'N/A')}")
    logger.info(f"   Perlu retrain : {'YA — ' + reason if should else 'Tidak'}")
    logger.info(f"   Total retrain : {sched_status.get('retrain_count', 0)}x")
    logger.info(f"{'='*55}")


# ======================================================================
# MODE: scheduler (loop retrain otomatis)
# ======================================================================
def mode_scheduler(cfg: dict, args):
    """
    Loop retrain otomatis.
    Cek setiap jam apakah retrain diperlukan, jalankan jika ya.
    """
    import time
    from monitoring.metrics import MetricsTracker, DriftDetector, AlertManager, RetrainScheduler
    from models.trainer import TrainingPipeline

    symbol    = args.symbol or cfg["data"]["primary_symbol"]
    tracker   = MetricsTracker(cfg)
    detector  = DriftDetector(cfg)
    alerter   = AlertManager(cfg)
    scheduler = RetrainScheduler(cfg)

    logger.info("🔁 Scheduler retrain otomatis berjalan... (Ctrl+C untuk berhenti)")

    while True:
        try:
            live_metrics = tracker.compute_live_performance(symbol)
            trades_df    = tracker.load_trades(symbol)
            drift        = detector.check(trades_df, baseline_accuracy=0.58) if not trades_df.empty else None
            alerter.check_all(live_metrics or {}, drift)

            should, reason = scheduler.should_retrain(live_metrics, drift)

            if should:
                logger.info(f"🔄 Memulai retrain otomatis: {reason}")
                try:
                    pipeline = TrainingPipeline(cfg)
                    results  = pipeline.run(symbol=symbol, from_file=False, skip_lstm=False)
                    tracker.record_training(symbol, results.get("backtest", {}))
                    scheduler.mark_retrained()
                    logger.info("✅ Retrain otomatis selesai.")
                except Exception as e:
                    logger.error(f"Retrain gagal: {e}")
            else:
                logger.info(f"⏳ Retrain belum diperlukan. Cek berikutnya dalam 1 jam.")

        except KeyboardInterrupt:
            logger.info("Scheduler dihentikan.")
            break
        except Exception as e:
            logger.error(f"Error di scheduler: {e}")

        # Tunggu 1 jam
        time.sleep(3600)


# ======================================================================
# CLI entry point
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML Forex Advisor — MT5 Integration System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mode", required=True,
        choices=["collect", "train", "backtest", "ticktest", "predict", "api", "monitor", "scheduler"],
        help=(
            "collect   : ambil data OHLCV dari MT5\n"
            "train     : training model ML\n"
            "backtest  : jalankan backtest dengan model tersimpan\n"
            "ticktest  : jalankan tick replay test dengan data real/dummy\n"
            "predict   : cetak prediksi sinyal saat ini\n"
            "api       : jalankan Flask API server\n"
            "monitor   : tampilkan laporan monitoring\n"
            "scheduler : loop retrain otomatis"
        ),
    )
    p.add_argument("--symbol",    type=str,  default=None, help="Simbol trading, mis. EURUSD")
    p.add_argument("--timeframe", type=str,  default=None, help="Timeframe: M5, M15, H1, H4, D1")
    p.add_argument("--bars",      type=int,  default=None, help="Jumlah bar (mode collect)")
    p.add_argument("--from-file", action="store_true",    help="Pakai data CSV yang ada (mode train)")
    p.add_argument("--skip-lstm", action="store_true",    help="Skip training LSTM (lebih cepat)")
    p.add_argument("--tick-file", type=str, default=None, help="Path ke file tick CSV untuk mode ticktest")
    p.add_argument("--tick-count", type=int, default=None, help="Jumlah tick dummy untuk generate jika file tidak ada. Untuk M15 gunakan jauh lebih banyak tick (auto-rekomendasi aktif).")
    p.add_argument("--tick-timeframe", type=str, default=None, help="Timeframe agregasi bar untuk ticktest")
    p.add_argument("--start-date", type=str, default=None, help="Tanggal mulai untuk tick backtest (format: YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, default=None, help="Tanggal akhir untuk tick backtest (format: YYYY-MM-DD)")
    p.add_argument("--config",    type=str,  default="config/config.yaml", help="Path config YAML")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg.get("paths", {}).get("logs_dir", "logs"))

    logger.info(f"🤖 ML Forex Advisor | Mode: {args.mode.upper()}")

    dispatch = {
        "collect":   mode_collect,
        "train":     mode_train,
        "backtest":  mode_backtest,
        "ticktest":  mode_ticktest,
        "predict":   mode_predict,
        "api":       mode_api,
        "monitor":   mode_monitor,
        "scheduler": mode_scheduler,
    }

    try:
        dispatch[args.mode](cfg, args)
    except KeyboardInterrupt:
        logger.info("Dihentikan oleh pengguna.")
    except Exception as e:
        logger.exception(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
