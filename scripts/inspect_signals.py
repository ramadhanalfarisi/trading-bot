import yaml
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make project root importable for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ensemble import EnsemblePredictor
from data.feature_engineering import FeatureEngineer
from data.preprocessor import Preprocessor
from models.xgboost_model import XGBoostTrainer
from models.lstm_model import LSTMTrainer
from data.collector import MT5Collector

cfg = yaml.safe_load(open('config/config.yaml'))

symbol = cfg['data']['primary_symbol']
frame = cfg['data']['timeframes']['primary']

# Load raw
collector = MT5Collector(cfg)
try:
    df_raw = collector.load_from_file(symbol, frame)
except FileNotFoundError:
    collector.connect()
    df_raw = collector.get_ohlcv(symbol, frame, cfg['data']['bars_history'])
    collector.disconnect()

# Trim to inference window
n_bars = cfg['data'].get('bars_inference', 300)
df_raw = df_raw.iloc[-n_bars:]

# Feature build
fe = FeatureEngineer(cfg)
df_feat = fe.build_features(df_raw, add_labels=False)

# Preprocessor + models
pre = Preprocessor(cfg)
pre.load_scaler()
feat_cols = pre.feature_cols
xgb = XGBoostTrainer(cfg)
xgb.load_model()
ensemble = EnsemblePredictor(cfg)
ensemble.load()

# Prepare inputs
X_all = pre.transform(df_feat[feat_cols])

# LSTM
seq_len = cfg['lstm']['sequence_length']
try:
    lstm = LSTMTrainer(cfg)
    lstm.build_model(len(feat_cols))
    lstm.load_model()
    Xs = []
    for i in range(seq_len, len(X_all)):
        Xs.append(X_all[i - seq_len:i])
    X_seq = np.array(Xs, dtype=np.float32)
    lstm_proba = lstm.predict_proba(X_seq)
    n = len(lstm_proba)
    xgb_proba_aligned = xgb.predict_proba(X_all)[-n:]
except Exception as e:
    xgb_proba = xgb.predict_proba(X_all)
    lstm_proba = xgb_proba
    xgb_proba_aligned = xgb_proba

results = ensemble.predict(lstm_proba, xgb_proba_aligned)
if isinstance(results, dict):
    results = [results]

sigs = [r['signal'] for r in results]
confs = [r['confidence'] for r in results]

print(f"Total samples: {len(results)}")
print(f"Signals count: BUY={sigs.count('BUY')}, SELL={sigs.count('SELL')}, HOLD={sigs.count('HOLD')}")
print(f"Confidence: min={min(confs):.4f}, max={max(confs):.4f}, mean={np.mean(confs):.4f}")

# show small sample
from collections import Counter
print('Top confidences sample:')
for r in results[:10]:
    print(r)

# histogram
import math
bins = [0.0, 0.5, 0.6, 0.7, 1.0]
counts = [0]* (len(bins)-1)
for c in confs:
    for i in range(len(bins)-1):
        if bins[i] <= c < bins[i+1]:
            counts[i]+=1
            break

print('Confidence bins (0-0.5,0.5-0.6,0.6-0.7,0.7-1.0):', counts)

# If too many HOLDs, print suggestion
hold_ratio = sigs.count('HOLD')/len(sigs) if sigs else 1
print(f'HOLD ratio: {hold_ratio:.3f}')
if hold_ratio > 0.9:
    print('Recommendation: lower ensemble confidence_threshold (config -> ensemble.confidence_threshold)')
