"""
models/trainer.py
Orkestrasi pipeline training lengkap: data → features → train → save.
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
from models.backtest import Backtester


class TrainingPipeline:
    """Pipeline training end-to-end."""

    def __init__(self, config: dict):
        self.config = config
        self.collector   = MT5Collector(config)
        self.feat_eng    = FeatureEngineer(config)
        self.preprocessor = Preprocessor(config)
        self.lstm_trainer = LSTMTrainer(config)
        self.xgb_trainer  = XGBoostTrainer(config)
        self.ensemble     = EnsemblePredictor(config)
        self.backtester   = Backtester(config)

    # ------------------------------------------------------------------
    # Run full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str = None,
        from_file: bool = False,
        skip_lstm: bool = False,
    ) -> dict:
        """
        Jalankan pipeline training lengkap.

        Args:
            symbol: simbol yang akan ditrain
            from_file: gunakan data CSV yang sudah tersimpan
            skip_lstm: skip training LSTM (hanya XGBoost)

        Returns:
            dict hasil metrik
        """
        symbol = symbol or self.config["data"]["primary_symbol"]
        tf     = self.config["data"]["timeframes"]["primary"]
        n_bars = self.config["data"]["bars_history"]

        logger.info(f"{'='*60}")
        logger.info(f"🚀 Training Pipeline | {symbol} {tf} | {n_bars} bar")
        logger.info(f"{'='*60}")

        # ① Ambil data
        if from_file:
            df_raw = self.collector.load_from_file(symbol, tf)
        else:
            self.collector.connect()
            df_raw = self.collector.get_ohlcv(symbol, tf, n_bars)
            self.collector.disconnect()

        # ② Feature engineering
        df_feat = self.feat_eng.build_features(df_raw, add_labels=True)
        feat_cols = self.feat_eng.get_feature_columns(df_feat)
        logger.info(f"Jumlah fitur: {len(feat_cols)}")

        # Simpan processed data
        processed_dir = Path(self.config["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(processed_dir / f"{symbol}_{tf}_features.csv")

        # ③ Preprocessing & split
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split(df_feat, feat_cols)
        logger.info(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

        results = {}

        # ④ Training XGBoost
        logger.info("\n--- Training XGBoost ---")
        xgb_metrics = self.xgb_trainer.train(X_train, y_train, X_val, y_val)
        results["xgboost"] = xgb_metrics

        # Log feature importance
        top_feats = self.xgb_trainer.get_top_features(feat_cols, top_n=10)
        logger.info("Top 10 Feature Importances:")
        for feat, imp in top_feats:
            logger.info(f"  {feat:<35} {imp:.4f}")

        # ⑤ Training LSTM
        if not skip_lstm:
            seq_len = self.config["lstm"]["sequence_length"]
            X_train_seq, y_train_seq = self.preprocessor.make_sequences(X_train, y_train, seq_len)
            X_val_seq,   y_val_seq   = self.preprocessor.make_sequences(X_val,   y_val,   seq_len)

            logger.info("\n--- Training LSTM ---")
            lstm_history = self.lstm_trainer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            results["lstm"] = {"epochs": len(lstm_history["val_acc"]),
                               "best_val_acc": max(lstm_history["val_acc"])}

            # Export ONNX
            onnx_path = self.config["paths"]["onnx_model"]
            self.lstm_trainer.export_onnx(onnx_path, len(feat_cols), seq_len)

        # ⑥ Evaluasi ensemble pada test set
        logger.info("\n--- Evaluasi Ensemble pada Test Set ---")
        xgb_proba_test = self.xgb_trainer.predict_proba(X_test)

        if not skip_lstm:
            X_test_seq, y_test_seq = self.preprocessor.make_sequences(X_test, y_test, seq_len)
            lstm_proba_test = self.lstm_trainer.predict_proba(X_test_seq)
            # Align: LSTM menggunakan seq_len lebih sedikit sample
            n_seq = len(y_test_seq)
            xgb_proba_aligned = xgb_proba_test[-n_seq:]
            y_test_aligned = y_test_seq

            ensemble_results = self.ensemble.predict(lstm_proba_test, xgb_proba_aligned)
        else:
            ensemble_results = self.ensemble.predict(xgb_proba_test, xgb_proba_test)
            y_test_aligned = y_test

        preds = [r["signal_id"] for r in ensemble_results]
        acc = np.mean(np.array(preds) == y_test_aligned)
        logger.info(f"Ensemble Test Accuracy: {acc:.4f}")
        results["ensemble_test_accuracy"] = acc

        # ⑦ Backtest pada test set
        logger.info("\n--- Backtest ---")
        test_df = df_feat.iloc[int(len(df_feat) * (self.config["preprocessing"]["train_ratio"] +
                                                     self.config["preprocessing"]["val_ratio"])):]
        test_df = test_df.iloc[-len(y_test_aligned):]

        bt_results = self.backtester.run(
            df=test_df,
            signals=[r["signal"] for r in ensemble_results],
        )
        results["backtest"] = bt_results

        # ⑧ Simpan ensemble config
        self.ensemble.save(meta={"symbol": symbol, "timeframe": tf, "n_features": len(feat_cols)})

        logger.info(f"\n{'='*60}")
        logger.info("✅ Training Pipeline Selesai!")
        logger.info(f"  XGBoost Val Acc : {xgb_metrics['val_accuracy']:.4f}")
        if not skip_lstm:
            logger.info(f"  LSTM Best Val Acc: {results['lstm']['best_val_acc']:.4f}")
        logger.info(f"  Ensemble Test Acc: {acc:.4f}")
        logger.info(f"  Sharpe Ratio    : {bt_results.get('sharpe_ratio', 'N/A')}")
        logger.info(f"  Max Drawdown    : {bt_results.get('max_drawdown', 'N/A')}")
        logger.info(f"{'='*60}")
        return results
