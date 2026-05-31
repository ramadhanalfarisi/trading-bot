"""
models/lstm_model.py
Model LSTM berbasis PyTorch untuk prediksi sinyal trading.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from loguru import logger


# ======================================================================
# Arsitektur LSTM
# ======================================================================

class ForexLSTM(nn.Module):
    """
    LSTM + Attention untuk klasifikasi sinyal forex.
    Output: 3 kelas (HOLD=0, BUY=1, SELL=2)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Attention
        self.attention = nn.Linear(hidden_size, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Attention over sequence
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (batch, seq, 1)
        context = (attn_weights * out).sum(dim=1)                 # (batch, hidden)
        logits = self.classifier(context)
        return logits


# ======================================================================
# Trainer
# ======================================================================

class LSTMTrainer:
    """Melatih dan mengelola model LSTM."""

    def __init__(self, config: dict):
        self.cfg = config["lstm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(config["paths"]["lstm_model"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: ForexLSTM = None
        logger.info(f"Device: {self.device}")

    def build_model(self, input_size: int) -> ForexLSTM:
        self.model = ForexLSTM(
            input_size=input_size,
            hidden_size=self.cfg["hidden_size"],
            num_layers=self.cfg["num_layers"],
            num_classes=3,
            dropout=self.cfg["dropout"],
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model LSTM: {n_params:,} parameter | Input: {input_size}")
        return self.model

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
    ) -> dict:
        """
        Latih LSTM dengan early stopping.

        Args:
            X_train/X_val: shape (n, seq_len, n_features)
            y_train/y_val: shape (n,) — label int 0/1/2

        Returns:
            dict history (train_loss, val_loss, val_acc per epoch)
        """
        if self.model is None:
            self.build_model(X_train.shape[2])

        # Dataset & DataLoader
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg["batch_size"], shuffle=False)

        # Hitung class weights untuk class imbalance
        counts = np.bincount(y_train, minlength=3)
        weights = torch.FloatTensor(1.0 / (counts + 1)).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(1, self.cfg["epochs"] + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # --- Validate ---
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{self.cfg['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_count = 0
                self._save_model()
            else:
                patience_count += 1
                if patience_count >= self.cfg["patience"]:
                    logger.info(f"⏹ Early stopping di epoch {epoch}")
                    break

        # Muat model terbaik
        self.load_model()
        logger.info(f"✅ Training selesai | Best Val Loss: {best_val_loss:.4f}")
        return history

    def _evaluate(self, loader: DataLoader, criterion) -> tuple:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * len(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        return total_loss / total, correct / total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Kembalikan probabilitas kelas (n_samples, 3)."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"✅ LSTM model dimuat dari {self.model_path}")

    def export_onnx(self, onnx_path: str, input_size: int, seq_len: int = 60):
        """Export model ke format ONNX untuk digunakan di MQL5."""
        import torch.onnx
        self.model.eval()
        dummy = torch.randn(1, seq_len, input_size).to(self.device)
        try:
            torch.onnx.export(
                self.model,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        except ModuleNotFoundError as exc:
            logger.warning(
                "ONNX export skipped: missing dependency '%s'. "
                "Install onnxscript with `pip install onnxscript` to enable ONNX export.",
                exc.name,
            )
            return
        logger.info(f"✅ Model ONNX disimpan ke {onnx_path}")
