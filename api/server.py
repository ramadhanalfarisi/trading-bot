"""
api/server.py
Flask REST API — menerima request dari MT5 Expert Advisor dan
mengembalikan sinyal prediksi ML.
"""
import yaml
import json
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger

from api.predictor import ForexPredictor

# ======================================================================
# Inisialisasi
# ======================================================================

def create_app(config: dict) -> Flask:
    app = Flask(__name__)
    CORS(app)

    predictor = ForexPredictor(config)

    # Muat model saat server start
    try:
        predictor.load_models()
    except Exception as e:
        logger.warning(f"Model belum tersedia: {e}. Jalankan training terlebih dahulu.")

    # Cache prediksi terakhir per symbol (hindari inferensi berulang di tick yang sama)
    prediction_cache = {}
    CACHE_TTL_SECONDS = 55  # Refresh tiap ~1 menit (sesuai timeframe M1)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": predictor._loaded,
        })

    # ------------------------------------------------------------------
    # Prediksi dari data yang dikirim EA (mode push)
    # ------------------------------------------------------------------

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        EA mengirim data OHLCV sebagai JSON, server mengembalikan sinyal.

        Request body:
        {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "bars": [
                {"time": "2024-01-01T00:00:00", "open": 1.08, "high": 1.082,
                 "low": 1.079, "close": 1.081, "volume": 1234},
                ...
            ]
        }
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body JSON diperlukan"}), 400

        symbol    = data.get("symbol", config["data"]["primary_symbol"])
        timeframe = data.get("timeframe", config["data"]["timeframes"]["primary"])
        bars      = data.get("bars", [])

        if len(bars) < 100:
            return jsonify({"error": f"Minimal 100 bar diperlukan, diterima: {len(bars)}"}), 400

        # Konversi ke DataFrame
        import pandas as pd
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Cek cache
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()
        if cache_key in prediction_cache:
            cached = prediction_cache[cache_key]
            if now - cached["ts"] < CACHE_TTL_SECONDS:
                result = cached["result"]
                result["cached"] = True
                return jsonify(result)

        try:
            result = predictor.predict_from_ohlcv(df, symbol)
            result["timestamp"] = datetime.utcnow().isoformat()
            result["cached"] = False
            prediction_cache[cache_key] = {"ts": now, "result": result}
            return jsonify(result)
        except Exception as e:
            logger.exception(f"Error saat prediksi: {e}")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # Prediksi live langsung dari MT5 (mode pull — server yang fetch data)
    # ------------------------------------------------------------------

    @app.route("/predict/live", methods=["GET"])
    def predict_live():
        """
        Server langsung mengambil data dari MT5 dan mengembalikan sinyal.
        Query params: symbol, timeframe
        """
        symbol    = request.args.get("symbol",    config["data"]["primary_symbol"])
        timeframe = request.args.get("timeframe", config["data"]["timeframes"]["primary"])

        try:
            result = predictor.predict_live(symbol, timeframe)
            result["timestamp"] = datetime.utcnow().isoformat()
            return jsonify(result)
        except Exception as e:
            logger.exception(f"Error live prediction: {e}")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # Info model
    # ------------------------------------------------------------------

    @app.route("/model/info", methods=["GET"])
    def model_info():
        info = {
            "models": {
                "lstm": predictor.use_lstm if predictor._loaded else False,
                "xgboost": predictor._loaded,
            },
            "config": {
                "ensemble_weights": {
                    "lstm": config["ensemble"]["lstm_weight"],
                    "xgboost": config["ensemble"]["xgboost_weight"],
                },
                "confidence_threshold": config["ensemble"]["confidence_threshold"],
                "risk_per_trade": config["risk"]["max_risk_per_trade"],
            },
        }
        if predictor._loaded:
            info["n_features"] = len(predictor.preprocessor.feature_cols)
        return jsonify(info)

    # ------------------------------------------------------------------
    # Force retrain (panggil dari scheduler atau manual)
    # ------------------------------------------------------------------

    @app.route("/retrain", methods=["POST"])
    def retrain():
        """Trigger retrain model secara async."""
        import threading
        from models.trainer import TrainingPipeline

        symbol = request.args.get("symbol", config["data"]["primary_symbol"])

        def _retrain():
            logger.info(f"🔄 Memulai retrain untuk {symbol}...")
            pipeline = TrainingPipeline(config)
            results = pipeline.run(symbol=symbol, from_file=False, skip_lstm=True)
            logger.info(f"✅ Retrain selesai: {results}")
            predictor.load_models()  # reload model baru

        t = threading.Thread(target=_retrain, daemon=True)
        t.start()
        return jsonify({"status": "retrain dimulai", "symbol": symbol})

    return app


# ======================================================================
# Entry point
# ======================================================================

def run_server(config: dict):
    app = create_app(config)
    host = config["api"]["host"]
    port = config["api"]["port"]
    debug = config["api"]["debug"]

    logger.info(f"🌐 API Server berjalan di http://{host}:{port}")
    logger.info(f"   Endpoint: POST /predict | GET /predict/live | GET /health")

    if debug:
        app.run(host=host, port=port, debug=True)
    else:
        from waitress import serve
        serve(app, host=host, port=port)


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_server(cfg)
