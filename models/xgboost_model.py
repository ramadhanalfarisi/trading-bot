"""
models/xgboost_model.py
Model XGBoost untuk prediksi sinyal trading forex.
"""
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger


class XGBoostTrainer:
    """Melatih dan mengelola model XGBoost."""

    CLASS_LABELS = [0, 1, 2]
    TARGET_NAMES = ["HOLD", "BUY", "SELL"]

    def __init__(self, config: dict):
        self.cfg = config["xgboost"]
        self.model_path = Path(config["paths"]["xgb_model"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: xgb.XGBClassifier = None
        self.feature_importances_: np.ndarray = None

    def build_model(self) -> xgb.XGBClassifier:
        device = "cuda" if self.cfg.get("use_gpu") else "cpu"
        self.model = xgb.XGBClassifier(
            n_estimators=self.cfg["n_estimators"],
            max_depth=self.cfg["max_depth"],
            learning_rate=self.cfg["learning_rate"],
            subsample=self.cfg["subsample"],
            colsample_bytree=self.cfg["colsample_bytree"],
            min_child_weight=self.cfg["min_child_weight"],
            gamma=self.cfg["gamma"],
            reg_alpha=self.cfg["reg_alpha"],
            reg_lambda=self.cfg["reg_lambda"],
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            device=device,
            random_state=42,
            n_jobs=-1,
        )
        return self.model

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
    ) -> dict:
        """
        Latih XGBoost dengan early stopping.

        Returns:
            dict dengan metrik evaluasi
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Training XGBoost | Train: {len(X_train)} | Val: {len(X_val)}")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        self.feature_importances_ = self.model.feature_importances_
        val_preds = self.model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        present_labels = sorted(set(np.concatenate([y_val, val_preds]).tolist()))
        present_names = [self.TARGET_NAMES[label] for label in present_labels]
        report = classification_report(
            y_val,
            val_preds,
            labels=present_labels,
            target_names=present_names,
            output_dict=True,
            zero_division=0,
        )

        logger.info(f"✅ XGBoost selesai | Val Acc: {acc:.4f} | Best iteration: {self.model.best_iteration}")
        logger.info("\n" + classification_report(
            y_val,
            val_preds,
            labels=present_labels,
            target_names=present_names,
            zero_division=0,
        ))

        self._save_model()
        return {"val_accuracy": acc, "classification_report": report}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Kembalikan probabilitas kelas (n_samples, 3)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_top_features(self, feature_names: list, top_n: int = 20) -> list:
        """Kembalikan N fitur terpenting."""
        if self.feature_importances_ is None:
            return []
        idx = np.argsort(self.feature_importances_)[::-1][:top_n]
        return [(feature_names[i], self.feature_importances_[i]) for i in idx]

    def _save_model(self):
        self.model.save_model(str(self.model_path))
        logger.info(f"💾 XGBoost disimpan ke {self.model_path}")

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
        if self.model is None:
            self.build_model()
        self.model.load_model(str(self.model_path))
        logger.info(f"✅ XGBoost dimuat dari {self.model_path}")

    def walk_forward_eval(self, X: np.ndarray, y: np.ndarray, preprocessor) -> dict:
        """Evaluasi model dengan walk-forward validation."""
        from sklearn.metrics import accuracy_score
        all_preds, all_true = [], []

        for fold_i, (X_tr, y_tr, X_te, y_te) in enumerate(preprocessor.walk_forward_splits(X, y)):
            temp_model = xgb.XGBClassifier(**self.model.get_params())
            temp_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds = temp_model.predict(X_te)
            acc = accuracy_score(y_te, preds)
            logger.info(f"  Walk-forward fold {fold_i+1}: acc={acc:.4f} | n_train={len(X_tr)}")
            all_preds.extend(preds)
            all_true.extend(y_te)

        overall_acc = accuracy_score(all_true, all_preds)
        logger.info(f"✅ Walk-forward overall accuracy: {overall_acc:.4f}")
        return {"walkforward_accuracy": overall_acc}
