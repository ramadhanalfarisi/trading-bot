# Skema Testing Model ML dengan Data Real Tick

Dokumen ini menjelaskan arsitektur dan alur testing model ML menggunakan data tick nyata dari MetaTrader 5.

## 1. Tujuan

- Evaluasi model ML dengan data tick real-time atau historis.
- Validasi kapan sinyal terbentuk pada level tick, bukan hanya pada bar OHLC.
- Ukur performa model dengan metrik trading nyata: PnL, win rate, drawdown, slippage, dan execution latency.

## 2. Komponen Utama

1. `data/collector.py`
   - `MT5Collector.get_latest_tick(symbol)` sudah tersedia untuk live tick.
   - Perlu ditambahkan kemampuan koleksi tick historis / replay tick apabila dibutuhkan.

2. `models/backtest.py`
   - Backtester saat ini berjalan pada data bar OHLC.
   - Untuk testing tick, perlu ditambahkan modul `TickBacktester` atau perluasan `Backtester`.

3. `data/feature_engineering.py`
   - Gunakan fitur existing untuk bar OHLC.
   - Untuk tick testing, tambahkan fitur mikrostruktural (spread, tick return, bid-ask imbalance, volume tick).

4. `api/predictor.py`
   - Sudah menaruh arsitektur inference live.
   - Bisa digunakan untuk validasi forward test terhadap tick stream.

5. `tests/test_all.py`
   - Tambahkan unit/integration test untuk koleksi tick dan replay tick.

## 3. Alur Testing Tick Data

### A. Data Acquisition

1. Koneksikan ke MT5 dengan `MT5Collector.connect()`.
2. Ambil tick terakhir secara terus-menerus via `get_latest_tick(symbol)`.
3. Simpan stream tick ke CSV, format minimal:
   - `time`, `bid`, `ask`, `volume` (opsional), `symbol`
4. Jika perlu, kumpulkan tick historis dari `mt5.copy_ticks_from(...)` atau `copy_ticks_range(...)`.

### B. Preprocessing Tick

1. Normalisasi waktu tick jika perlu (`UTC` / timezone konsisten).
2. Hitung nilai berikut per tick atau per agregasi mikro:
   - bid-ask spread
   - mid-price: `(bid+ask)/2`
   - tick return / price change
   - moving average tick level (short window)
   - volume tick / tick count
3. Pilih dua mode testing:
   - `tick-to-bar`: agregasi tick ke timeframe seperti M1 / M5 lalu pakai pipeline existing.
   - `tick-native`: gunakan tick features langsung untuk model.

### C. Feature Engineering & Model Input

1. Untuk tick-native:
   - Siapkan layer fitur khusus tick di `data/feature_engineering.py`.
   - Ekstrak fitur teknikal sekilas per tick (VWAP, spread momentum, mikro order flow).
2. Untuk tick-to-bar:
   - Gunakan `OHLC` bar hasil agregasi tick.
   - Ambil fitur existing dari `FeatureEngineer`.

### D. Model Inference dan Label

1. Gunakan model terlatih untuk memprediksi sinyal pada `tick time` atau pada akhir bar.
2. Hitung label sebenarnya berdasarkan pergerakan harga di horizon berikutnya (misalnya 30 tick / 1 bar ke depan).
3. Simulasi target label bisa berupa:
   - `BUY` jika close kena target naik pada horizon berikutnya.
   - `SELL` jika turun.
   - `HOLD` jika tidak memenuhi ambang threshold.

### E. Tick Replay Backtest

1. Jalankan tick replay sebagai loop:
   - untuk setiap tick `t`, hitung fitur, buat prediksi, buka/kelola trade.
2. Perhatikan execution model:
   - entry: gunakan next tick bid/ask sebagai harga eksekusi.
   - exit SL/TP: cek tick-level high/low atau quote bid/ask.
   - spread: gunakan `ask` untuk BUY, `bid` untuk SELL.
3. Hitung PnL berdasarkan tick-level exit.
4. Keluarkan trade log:
   - `entry_time`, `entry_price`, `exit_time`, `exit_price`, `pnl`, `direction`, `confidence`.

## 4. Metrik Evaluasi

- Total trades
- Win rate
- Profit factor
- Sharpe ratio
- Max drawdown
- Total PnL
- Average trade duration
- Slippage rata-rata (entry/exit vs pergerakan target)
- Hit rate TP/SL
- Realtime accuracy untuk prediction horizon

## 5. Skenario Test yang Direkomendasikan

### 1. Historical Tick Replay

- Input: file CSV tick historis
- Output: laporan backtest tick-level + trade log
- Kegunaan: validasi strategi tanpa risiko produksi

### 2. Live Forward Validation

- Input: stream tick real-time dari MT5
- Output: sinyal prediksi & realisasi harga horizon berikutnya
- Kegunaan: cek drift model dan latency

### 3. End-to-end API + EA Simulation

- Jalankan `api/server.py` dan `mt5_ea/MLAdvisor.mq5`
- Validasi bahwa prediksi live ditangkap dengan benar oleh EA
- Hitung outcome berdasarkan quote tick riil dan order fill

## 6. Rekomendasi Implementasi di Repo Ini

1. Tambahkan method baru di `data/collector.py`:
   - `collect_ticks(symbol, n_ticks, save=True)`
   - `load_tick_history(symbol, filename)`

2. Tambahkan modul baru:
   - `models/tick_backtest.py`
   - Atau perluas `models/backtest.py` dengan `TickBacktester`

3. Tambahkan pipeline testing di `main.py`:
   - `--mode ticktest`
   - `--tick-file <csv>`
   - `--tick-horizon <n>`

4. Tambahkan dokumentasi testing di README atau file ini.

## 7. Contoh Alur `TickBacktester`

1. Load tick history
2. Agregasi ke bar jika diperlukan
3. Hitung fitur untuk setiap bar/tick
4. Prediksi sinyal
5. Buka posisi berdasarkan `ask`/`bid`
6. Monitor SL/TP per tick
7. Tutup posisi di akhir replay jika masih terbuka
8. Hitung metrik

## 8. Contoh Config yang Disarankan

```yaml
data:
  tick_dir: data/tick
  tick_symbols:
    - EURUSD
    - GBPUSD
  tick_batch: 100000

testing:
  tick_horizon: 30
  tick_spread_mode: real
  tick_label_threshold: 0.00015

risk:
  max_risk_per_trade: 0.01
  sl_atr_multiplier: 1.5
  tp_atr_multiplier: 2.5
```

---

Dengan skema ini, Anda bisa mulai mengimplementasikan testing tick nyata secara bertahap:

1. Kumpulkan data tick nyata.
2. Bangun replay pipeline tick.
3. Sambungkan ke model inference dan backtester.
4. Validasi output dengan metrik trading riil.

Jika Anda ingin, saya bisa lanjut membuat kode skeleton `tick_backtest.py` dan update `main.py` untuk mode `ticktest`.
