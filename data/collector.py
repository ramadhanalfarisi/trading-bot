"""
data/collector.py
Mengambil data OHLCV dan tick dari MetaTrader 5.
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 library tidak terinstall. Gunakan data dummy untuk testing.")

# Mapping nama timeframe string ke konstanta MT5
TIMEFRAME_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
    "W1":  32769,
    "MN1": 49153,
}


class MT5Collector:
    """Kelas untuk mengambil data dari MetaTrader 5."""

    def __init__(self, config: dict):
        self.config = config
        self.connected = False
        self.data_dir = Path(config["data"].get("data_dir", "data/raw"))
        self.tick_dir = Path(config["data"].get("tick_dir", "data/tick"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tick_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Koneksi
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Inisialisasi dan login ke MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 tidak tersedia — mode simulasi aktif.")
            self.connected = False
            return False

        mt5_cfg = self.config["mt5"]
        if not mt5.initialize():
            logger.error(f"MT5 initialize() gagal: {mt5.last_error()}")
            return False

        login_ok = mt5.login(
            login=mt5_cfg["login"],
            password=mt5_cfg["password"],
            server=mt5_cfg["server"],
        )
        if not login_ok:
            logger.error(f"MT5 login gagal: {mt5.last_error()}")
            mt5.shutdown()
            return False

        info = mt5.account_info()
        logger.info(f"✅ Terhubung ke MT5 | Akun: {info.login} | Balance: {info.balance:.2f} {info.currency}")
        self.connected = True
        return True

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected.")

    # ------------------------------------------------------------------
    # Ambil data OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        timeframe_str: str,
        n_bars: int,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Ambil N bar terakhir OHLCV dari MT5.

        Returns:
            DataFrame dengan kolom: open, high, low, close, volume
        """
        if not self.connected:
            logger.warning("Tidak terhubung ke MT5 — menggunakan data dummy.")
            return self._generate_dummy_data(n_bars)

        tf_code = TIMEFRAME_MAP.get(timeframe_str.upper())
        if tf_code is None:
            raise ValueError(f"Timeframe '{timeframe_str}' tidak valid. Pilihan: {list(TIMEFRAME_MAP.keys())}")

        # Pastikan symbol tersedia
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} tidak ditemukan di MT5.")

        rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Gagal mengambil data {symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.sort_index(inplace=True)

        logger.info(f"📊 Data {symbol} {timeframe_str}: {len(df)} bar | {df.index[0]} → {df.index[-1]}")

        if save:
            fname = self.data_dir / f"{symbol}_{timeframe_str}.csv"
            df.to_csv(fname)
            logger.info(f"💾 Disimpan ke {fname}")

        return df

    # ------------------------------------------------------------------
    # Ambil info akun dan posisi terbuka
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Kembalikan info akun MT5 sebagai dict."""
        if not self.connected:
            return {"balance": 10000, "equity": 10000, "margin_free": 9000, "currency": "USD"}
        info = mt5.account_info()
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    def get_open_positions(self, symbol: str = None) -> list:
        """Kembalikan posisi terbuka, opsional filter per symbol."""
        if not self.connected:
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def get_symbol_info(self, symbol: str) -> dict:
        """Ambil info symbol (point, digits, spread, dll)."""
        if not self.connected:
            return {"point": 0.00001, "digits": 5, "spread": 1, "volume_min": 0.01}
        info = mt5.symbol_info(symbol)
        return {
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
        }

    def get_latest_tick(self, symbol: str) -> dict:
        """Ambil harga terakhir (bid/ask)."""
        if not self.connected:
            return {"bid": 1.08500, "ask": 1.08502, "time": datetime.now()}
        tick = mt5.symbol_info_tick(symbol)
        return {"bid": tick.bid, "ask": tick.ask, "time": datetime.fromtimestamp(tick.time)}

    # ------------------------------------------------------------------
    # Ambil data tick
    # ------------------------------------------------------------------

    def collect_ticks(
        self,
        symbol: str,
        n_ticks: int = 5000,
        save: bool = True,
    ) -> pd.DataFrame:
        """Ambil tick history dari MT5 atau generate dummy jika offline."""
        if not self.connected:
            logger.warning("Tidak terhubung ke MT5 — menggunakan tick dummy untuk testing.")
            df = self._generate_dummy_tick_history(symbol, n_ticks)
        else:
            now = datetime.now(timezone.utc)
            from_time = now - timedelta(hours=1)
            ticks = mt5.copy_ticks_from(symbol, from_time, n_ticks, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                logger.warning("Gagal mengambil tick MT5 — menggunakan tick dummy.")
                df = self._generate_dummy_tick_history(symbol, n_ticks)
            else:
                df = pd.DataFrame(ticks)
                df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
                df = df.rename(columns={"time": "time", "bid": "bid", "ask": "ask", "volume": "volume"})
                df = df[["time", "bid", "ask", "volume"]]
                df.set_index("time", inplace=True)
                df.sort_index(inplace=True)

        if save:
            fname = self.tick_dir / f"{symbol}_ticks.csv"
            df.to_csv(fname)
            logger.info(f"💾 Tick history disimpan ke {fname}")

        return df

    def collect_ticks_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Ambil tick history dari MT5 dalam date range tertentu.
        
        Args:
            symbol: Simbol trading (e.g., 'EURUSD')
            start_date: Tanggal mulai (format: 'YYYY-MM-DD')
            end_date: Tanggal akhir (format: 'YYYY-MM-DD')
            save: Simpan ke file CSV
            
        Returns:
            DataFrame dengan kolom: bid, ask, volume
        """
        try:
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True)
        except Exception as e:
            logger.error(f"Format date tidak valid: {e}")
            raise ValueError(f"Format date harus YYYY-MM-DD, got {start_date} dan {end_date}")

        logger.info(f"📥 Mengambil tick history {symbol} | {start_date} to {end_date}")

        if not self.connected:
            logger.warning("Tidak terhubung ke MT5 — menggunakan tick dummy untuk testing.")
            df = self._generate_dummy_tick_history_range(symbol, start_dt, end_dt)
        else:
            try:
                # MT5 API untuk mengambil tick dalam range
                ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_ALL)
                if ticks is None or len(ticks) == 0:
                    logger.warning(f"Tidak ada tick untuk {symbol} di range {start_date} - {end_date}")
                    df = self._generate_dummy_tick_history_range(symbol, start_dt, end_dt)
                else:
                    df = pd.DataFrame(ticks)
                    df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
                    df = df.rename(columns={"bid": "bid", "ask": "ask"})
                    df = df[["time", "bid", "ask", "volume"]]
                    df.set_index("time", inplace=True)
                    df.sort_index(inplace=True)
                    logger.info(f"✅ Berhasil mengambil {len(df)} tick untuk {symbol}")
            except Exception as e:
                logger.warning(f"Error saat mengambil tick dari MT5: {e}. Menggunakan dummy data.")
                df = self._generate_dummy_tick_history_range(symbol, start_dt, end_dt)

        if save:
            fname = self.tick_dir / f"{symbol}_ticks.csv"
            df.to_csv(fname)
            logger.info(f"💾 Tick history disimpan ke {fname}")

        return df

    def load_tick_history(self, symbol: str, filename: str = None) -> pd.DataFrame:
        """Load tick history dari file CSV."""
        if filename:
            fname = Path(filename)
        else:
            fname = self.tick_dir / f"{symbol}_ticks.csv"
        if not fname.exists():
            raise FileNotFoundError(f"File tick tidak ditemukan: {fname}")

        df = pd.read_csv(fname)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        else:
            # If the CSV was saved with a DateTimeIndex, load it from the first column
            df.columns = [col.strip() for col in df.columns]
            idx_name = df.columns[0]
            try:
                df.index = pd.to_datetime(df[idx_name], utc=True, errors="coerce")
                df = df.drop(columns=[idx_name])
            except Exception:
                raise ValueError(f"CSV tick tidak berisi kolom waktu yang valid: {fname}")

        # Force the index name to be time so downstream reset_index() produces a proper time column
        df.index.name = "time"

        df.sort_index(inplace=True)
        logger.info(f"📂 Loaded tick history {len(df)} bar dari {fname}")
        return df

    def load_tick_history_filtered(
        self, 
        symbol: str, 
        filename: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Load tick history dari file CSV dan filter berdasarkan date range.
        
        Args:
            symbol: Simbol trading
            filename: Path ke file tick CSV (opsional)
            start_date: Tanggal mulai filter (format: 'YYYY-MM-DD')
            end_date: Tanggal akhir filter (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame dengan tick yang sudah difilter
        """
        df = self.load_tick_history(symbol, filename)
        
        if start_date is None and end_date is None:
            return df
        
        # Convert dates
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_dt]
        
        logger.info(f"Filtered tick history to {len(df)} ticks between {start_date} and {end_date}")
        return df

    def _generate_dummy_tick_history(self, symbol: str, n_ticks: int) -> pd.DataFrame:
        """Generate dummy tick history untuk testing tanpa MT5."""
        np.random.seed(42)
        start_price = 1.0850
        times = pd.date_range(end=datetime.now(timezone.utc), periods=n_ticks, freq="1s", tz="UTC")
        mid = start_price + np.cumsum(np.random.randn(n_ticks) * 0.00002)
        spread = np.abs(np.random.randn(n_ticks) * 0.00001) + 0.00001
        bids = mid - spread / 2
        asks = mid + spread / 2
        df = pd.DataFrame({
            "bid": bids,
            "ask": asks,
            "volume": np.random.randint(1, 10, size=n_ticks).astype(float),
        }, index=times)
        return df

    def _generate_dummy_tick_history_range(self, symbol: str, start_dt, end_dt) -> pd.DataFrame:
        """Generate dummy tick history dalam date range tertentu untuk testing tanpa MT5."""
        np.random.seed(42)
        # Estimate tick count: ~1 tick per second
        time_diff = end_dt - start_dt
        estimated_ticks = int(time_diff.total_seconds())
        
        # Cap at reasonable limits for demo
        n_ticks = min(estimated_ticks, 100_000)
        
        logger.info(f"Generating {n_ticks} dummy ticks dari {start_dt} hingga {end_dt}")
        
        times = pd.date_range(start=start_dt, end=end_dt, periods=n_ticks, tz="UTC")
        start_price = 1.0850
        mid = start_price + np.cumsum(np.random.randn(n_ticks) * 0.00002)
        spread = np.abs(np.random.randn(n_ticks) * 0.00001) + 0.00001
        bids = mid - spread / 2
        asks = mid + spread / 2
        df = pd.DataFrame({
            "bid": bids,
            "ask": asks,
            "volume": np.random.randint(1, 10, size=n_ticks).astype(float),
        }, index=times)
        return df

    # ------------------------------------------------------------------
    # Data dummy untuk testing tanpa MT5
    # ------------------------------------------------------------------

    def _generate_dummy_data(self, n_bars: int) -> pd.DataFrame:
        """Generate data OHLCV simulasi untuk testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="1h", tz="UTC")
        close = 1.08 + np.cumsum(np.random.randn(n_bars) * 0.0005)
        noise = np.abs(np.random.randn(n_bars) * 0.0002)

        df = pd.DataFrame({
            "open":   close - np.random.randn(n_bars) * 0.0001,
            "high":   close + noise,
            "low":    close - noise,
            "close":  close,
            "volume": np.random.randint(100, 5000, n_bars).astype(float),
        }, index=dates)
        return df

    # ------------------------------------------------------------------
    # Load dari file CSV (jika sudah dikumpulkan sebelumnya)
    # ------------------------------------------------------------------

    def load_from_file(self, symbol: str, timeframe_str: str) -> pd.DataFrame:
        fname = self.data_dir / f"{symbol}_{timeframe_str}.csv"
        if not fname.exists():
            raise FileNotFoundError(f"File {fname} tidak ditemukan. Jalankan collect terlebih dahulu.")
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        logger.info(f"📂 Loaded {len(df)} bar dari {fname}")
        return df


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    collector = MT5Collector(cfg)
    collector.connect()

    df = collector.get_ohlcv(
        symbol=cfg["data"]["primary_symbol"],
        timeframe_str=cfg["data"]["timeframes"]["primary"],
        n_bars=cfg["data"]["bars_history"],
    )
    print(df.tail())
    collector.disconnect()
