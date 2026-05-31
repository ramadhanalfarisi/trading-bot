"""
data/preprocessor.py
Normalisasi fitur dan split data train/val/test (walk-forward aware).
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from loguru import logger

SCALER_MAP = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


class Preprocessor:
    """Normalisasi dan pemisahan data untuk training ML."""

    def __init__(self, config: dict):
        self.cfg = config
        scaler_name = config["preprocessing"]["scaler"]
        self.scaler = SCALER_MAP.get(scaler_name, StandardScaler)()
        self.feature_cols: list = []
        self.scaler_path = Path(config["paths"]["scaler"])
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Split train / val / test (time-series safe — no shuffle)
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame, feature_cols: list):
        """
        Split DataFrame secara kronologis.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.feature_cols = feature_cols
        train_r = self.cfg["preprocessing"]["train_ratio"]
        val_r   = self.cfg["preprocessing"]["val_ratio"]

        n = len(df)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        train = df.iloc[:n_train]
        val   = df.iloc[n_train : n_train + n_val]
        test  = df.iloc[n_train + n_val :]

        logger.info(f"Split | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

        X_train = train[feature_cols].values
        X_val   = val[feature_cols].values
        X_test  = test[feature_cols].values
        y_train = train["label"].values.astype(int)
        y_val   = val["label"].values.astype(int)
        y_test  = test["label"].values.astype(int)

        # Fit scaler hanya pada train
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        self._save_scaler()
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # LSTM: buat sequence 3D (samples, timesteps, features)
    # ------------------------------------------------------------------

    def make_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
    ):
        """
        Ubah array 2D (n_samples, n_features) menjadi
        array 3D (n_sequences, seq_len, n_features) untuk LSTM.
        """
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    # ------------------------------------------------------------------
    # Walk-Forward Validation Generator
    # ------------------------------------------------------------------

    def walk_forward_splits(self, X: np.ndarray, y: np.ndarray):
        """
        Generator walk-forward split untuk validasi.
        Menghasilkan (X_train, y_train, X_test, y_test) per fold.
        """
        train_size = self.cfg["preprocessing"]["walkforward_train_size"]
        test_size  = self.cfg["preprocessing"]["walkforward_test_size"]
        n_splits   = self.cfg["preprocessing"]["walkforward_n_splits"]
        n = len(X)

        for i in range(n_splits):
            end_train = n - (n_splits - i) * test_size
            start_train = max(0, end_train - train_size)
            end_test = end_train + test_size

            if end_test > n:
                break

            yield (
                X[start_train:end_train],
                y[start_train:end_train],
                X[end_train:end_test],
                y[end_train:end_test],
            )

    # ------------------------------------------------------------------
    # Transform data baru (inferensi)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame fitur untuk inferensi (pakai scaler yang sudah fit)."""
        if not self.feature_cols:
            raise RuntimeError("Preprocessor belum di-fit. Panggil split() dulu atau load_scaler().")
        X = df[self.feature_cols].values
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Simpan / muat scaler
    # ------------------------------------------------------------------

    def _save_scaler(self):
        joblib.dump({"scaler": self.scaler, "feature_cols": self.feature_cols}, self.scaler_path)
        logger.info(f"💾 Scaler disimpan ke {self.scaler_path}")

    def load_scaler(self):
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler tidak ditemukan: {self.scaler_path}")
        data = joblib.load(self.scaler_path)
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        logger.info(f"✅ Scaler dimuat | {len(self.feature_cols)} fitur")
