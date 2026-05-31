# 🤖 Forex ML Advisor System — MT5 Integration

Sistem trading advisor berbasis Machine Learning yang terintegrasi dengan MetaTrader 5.

## 📁 Struktur Proyek

```
forex_ml_system/
├── config/
│   └── config.yaml              # Konfigurasi global
├── data/
│   ├── collector.py             # Ambil data dari MT5
│   ├── feature_engineering.py   # Buat fitur teknikal
│   └── preprocessor.py          # Normalisasi & split data
├── models/
│   ├── lstm_model.py            # Model LSTM (PyTorch)
│   ├── xgboost_model.py         # Model XGBoost
│   ├── ensemble.py              # Ensemble kedua model
│   ├── trainer.py               # Training pipeline
│   └── backtest.py              # Walk-forward backtest
├── api/
│   ├── server.py                # Flask REST API
│   └── predictor.py             # Inference engine
├── mt5_ea/
│   ├── MLAdvisor.mq5            # Expert Advisor MQL5
│   └── MLAdvisorONNX.mq5        # EA dengan ONNX langsung
├── monitoring/
│   ├── dashboard.py             # Streamlit dashboard
│   └── metrics.py               # Perhitungan metrik
├── tests/
│   └── test_all.py              # Unit tests
├── notebooks/
│   └── exploration.ipynb        # EDA notebook
├── main.py                      # Entry point utama
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi
Edit `config/config.yaml` sesuai akun MT5 dan preferensi Anda.

### 3. Kumpulkan Data
```bash
python main.py --mode collect --symbol EURUSD --timeframe H1 --bars 5000
```

### 4. Training Model
```bash
python main.py --mode train --symbol EURUSD
```

### 5. Backtest
```bash
python main.py --mode backtest --symbol EURUSD
```

### 6. Jalankan API Server
```bash
python main.py --mode api
```

### 7. Monitoring Dashboard
```bash
streamlit run monitoring/dashboard.py
```

### 8. Setup EA di MT5
- Copy file `mt5_ea/MLAdvisor.mq5` ke folder `MQL5/Experts/`
- Compile di MetaEditor
- Attach EA ke chart dengan symbol & timeframe yang sesuai
- Pastikan "Allow WebRequest" untuk URL `http://127.0.0.1:5000`

## ⚙️ Arsitektur Sistem

```
MT5 (live data) ──► Python Collector ──► Feature Engineering
                                                │
                                         Model Training (offline)
                                                │
                                    ONNX Export / Flask API
                                                │
MT5 EA ──► HTTP Request ──► Prediction ──► Risk Filter ──► OrderSend()
                                                │
                                         Monitoring Dashboard
```

## 📊 Model yang Digunakan

| Model | Kegunaan | Keunggulan |
|-------|----------|------------|
| LSTM | Pola temporal/sekuensial | Menangkap tren jangka panjang |
| XGBoost | Fitur tabular | Cepat, akurat untuk pola non-linear |
| Ensemble | Kombinasi keduanya | Lebih robust, mengurangi overfitting |

## ⚠️ Disclaimer

Sistem ini hanya untuk tujuan edukasi dan penelitian. Trading forex mengandung risiko tinggi. Selalu uji di akun demo sebelum live trading.
