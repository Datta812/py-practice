"""
Bharat Pashudhan App — AI Breed Identification Backend
======================================================
Flask REST API that wraps the vishnuamar/cattle-breed-classifier ONNX model.
Provides breed identification from uploaded animal images.

Requirements:
    pip install flask flask-cors onnxruntime huggingface_hub pillow torchvision numpy

Run:
    python app.py
    
API will be available at: http://localhost:5000
"""

import os
import io
import json
import time
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Optional imports (graceful degradation if not installed) ──
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed — running in DEMO mode")

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("torchvision not installed — using basic preprocessing")

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("huggingface_hub not installed — model auto-download disabled")


# ── App setup ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Allow frontend on any origin

# ── Config ──
MODEL_REPO   = "vishnuamar/cattle-breed-classifier"
MODEL_FILE   = "model.onnx"
PROTO_FILE   = "prototypes.json"
CACHE_DIR    = Path("./model_cache")
UPLOAD_DIR   = Path("./uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

CACHE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Breed metadata ──
BREED_METADATA = {
    "Gir":        {"type": "Cattle",  "subtype": "Indigenous", "origin": "Gujarat",          "purpose": "Dairy",        "milk_yield": "6–12 L/day",  "fat_pct": "4.5%"},
    "Sahiwal":    {"type": "Cattle",  "subtype": "Indigenous", "origin": "Punjab/Rajasthan", "purpose": "Dairy",        "milk_yield": "8–14 L/day",  "fat_pct": "4.8%"},
    "Ongole":     {"type": "Cattle",  "subtype": "Indigenous", "origin": "Andhra Pradesh",   "purpose": "Dual purpose", "milk_yield": "4–7 L/day",   "fat_pct": "4.2%"},
    "Kankrej":    {"type": "Cattle",  "subtype": "Indigenous", "origin": "Gujarat/Rajasthan","purpose": "Dual purpose", "milk_yield": "4–8 L/day",   "fat_pct": "4.0%"},
    "Tharparkar": {"type": "Cattle",  "subtype": "Indigenous", "origin": "Rajasthan",        "purpose": "Dual purpose", "milk_yield": "5–9 L/day",   "fat_pct": "4.6%"},
    "Murrah":     {"type": "Buffalo", "subtype": "Indigenous", "origin": "Haryana/Punjab",   "purpose": "Dairy",        "milk_yield": "10–18 L/day", "fat_pct": "7.5%"},
    "Surti":      {"type": "Buffalo", "subtype": "Indigenous", "origin": "Gujarat",          "purpose": "Dairy",        "milk_yield": "7–11 L/day",  "fat_pct": "8.5%"},
    "Jaffarbadi": {"type": "Buffalo", "subtype": "Indigenous", "origin": "Gujarat",          "purpose": "Dairy",        "milk_yield": "8–14 L/day",  "fat_pct": "7.8%"},
    "Mehsana":    {"type": "Buffalo", "subtype": "Crossbred",  "origin": "Gujarat",          "purpose": "Dairy",        "milk_yield": "8–13 L/day",  "fat_pct": "7.2%"},
    "Bhadawari":  {"type": "Buffalo", "subtype": "Indigenous", "origin": "UP/MP",            "purpose": "Dairy",        "milk_yield": "5–7 L/day",   "fat_pct": "11.0%"},
}

BUFFALO_BREEDS = {"Murrah", "Surti", "Jaffarbadi", "Mehsana", "Bhadawari"}


# ══════════════════════════════════════════
# MODEL MANAGER
# ══════════════════════════════════════════

class BreedClassifier:
    """Wraps the ONNX breed classifier with download, caching, and inference."""

    def __init__(self):
        self.session   = None
        self.prototypes = None
        self.transform  = None
        self.loaded     = False
        self.demo_mode  = False
        self._load()

    def _load(self):
        """Download model from HuggingFace and initialise ONNX session."""
        if not ONNX_AVAILABLE or not HF_AVAILABLE:
            logger.warning("Running in DEMO mode — no real inference")
            self.demo_mode = True
            self.loaded = True
            return

        model_path = CACHE_DIR / MODEL_FILE
        proto_path = CACHE_DIR / PROTO_FILE

        try:
            if not model_path.exists():
                logger.info("Downloading model from HuggingFace …")
                dl = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=str(CACHE_DIR))
                logger.info(f"Model saved → {dl}")
            else:
                logger.info(f"Using cached model: {model_path}")

            if not proto_path.exists():
                logger.info("Downloading prototypes …")
                hf_hub_download(MODEL_REPO, PROTO_FILE, local_dir=str(CACHE_DIR))

            # Load ONNX session
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(model_path), sess_opts=sess_opts)

            # Load prototypes
            with open(proto_path) as f:
                data = json.load(f)
            self.prototypes = data.get("prototypes", data)  # handle both formats

            # ImageNet preprocessing transform
            if TORCHVISION_AVAILABLE:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

            self.loaded = True
            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"Model load failed: {e} — falling back to DEMO mode")
            self.demo_mode = True
            self.loaded = True

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to model-ready numpy array."""
        img = pil_image.convert("RGB")

        if self.transform and TORCHVISION_AVAILABLE:
            import torch
            tensor = self.transform(img)
            return tensor.unsqueeze(0).numpy()

        # Fallback: manual preprocessing
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)          # HWC → CHW
        return arr[np.newaxis].astype(np.float32)  # add batch dim

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Run breed classification.
        Returns dict with breed, confidence, animal_type, all_scores.
        """
        if self.demo_mode:
            return self._demo_predict()

        t0 = time.time()

        # Preprocess
        inp = self._preprocess(pil_image)

        # ONNX inference — get feature vector
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: inp})[0][0]  # shape: (2048,)

        # Cosine similarity vs every breed prototype
        scores = {}
        for breed, proto in self.prototypes.items():
            scores[breed] = self._cosine_similarity(features, np.array(proto, dtype=np.float32))

        # Sort descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_breed, top_score = sorted_scores[0]

        # Normalise raw cosine scores to [0,1] for display
        vals = np.array([v for _, v in sorted_scores])
        vmin, vmax = vals.min(), vals.max()
        def norm(v):
            return float((v - vmin) / (vmax - vmin + 1e-8))

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        meta = BREED_METADATA.get(top_breed, {})

        return {
            "breed":       top_breed,
            "confidence":  round(norm(top_score), 4),
            "animal_type": "Buffalo" if top_breed in BUFFALO_BREEDS else "Cattle",
            "all_scores":  {b: round(norm(s), 4) for b, s in sorted_scores},
            "top_5":       [{"breed": b, "score": round(norm(s), 4)} for b, s in sorted_scores[:5]],
            "inference_ms": elapsed_ms,
            "metadata":    meta,
            "demo_mode":   False,
        }

    def _demo_predict(self) -> dict:
        """Return a plausible random prediction when model is unavailable."""
        import random
        breeds = list(BREED_METADATA.keys())
        top = random.choice(breeds)
        scores_raw = {b: random.random() for b in breeds}
        scores_raw[top] = 0.85 + random.random() * 0.12
        total = sum(scores_raw.values())
        scores = {b: round(v / total, 4) for b, v in scores_raw.items()}
        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        meta = BREED_METADATA.get(top, {})
        return {
            "breed":       top,
            "confidence":  scores[top],
            "animal_type": "Buffalo" if top in BUFFALO_BREEDS else "Cattle",
            "all_scores":  scores,
            "top_5":       [{"breed": b, "score": s} for b, s in sorted_s[:5]],
            "inference_ms": round(40 + 15 * random.random(), 1),
            "metadata":    meta,
            "demo_mode":   True,
        }


# Singleton — loaded once at startup
classifier = BreedClassifier()


# ══════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — useful for deployment checks."""
    return jsonify({
        "status":      "ok",
        "model_loaded": classifier.loaded,
        "demo_mode":   classifier.demo_mode,
        "breeds":      list(BREED_METADATA.keys()),
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/breeds", methods=["GET"])
def get_breeds():
    """Return full breed metadata for the frontend breed database."""
    return jsonify({
        "total":  len(BREED_METADATA),
        "breeds": [
            {"name": name, **meta}
            for name, meta in BREED_METADATA.items()
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Accepts multipart/form-data with field 'image' (file upload)
    OR application/json with field 'image_base64' (base64-encoded image).

    Returns JSON with breed prediction and confidence scores.
    """
    start = time.time()

    # ── Parse image ──
    pil_image = None

    # Case 1: file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.content_type not in ALLOWED_TYPES:
            return jsonify({"error": f"Unsupported file type: {file.content_type}"}), 400
        raw = file.read()
        if len(raw) > MAX_FILE_SIZE:
            return jsonify({"error": "File too large (max 10 MB)"}), 400
        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not open image: {e}"}), 400

    # Case 2: base64 in JSON
    elif request.is_json and "image_base64" in request.json:
        try:
            b64 = request.json["image_base64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]  # strip data:image/...;base64,
            raw = base64.b64decode(b64)
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {e}"}), 400

    else:
        return jsonify({"error": "No image provided. Send 'image' file or 'image_base64' JSON field."}), 400

    # ── Run inference ──
    try:
        result = classifier.predict(pil_image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    result["total_ms"] = round((time.time() - start) * 1000, 1)
    logger.info(f"Predicted: {result['breed']} ({result['confidence']*100:.1f}%) in {result['total_ms']}ms")
    return jsonify(result)


@app.route("/api/predict/demo", methods=["GET"])
def predict_demo():
    """Demo endpoint — returns a random prediction (no image needed). Useful for frontend testing."""
    result = classifier._demo_predict()
    result["total_ms"] = result["inference_ms"]
    return jsonify(result)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ══════════════════════════════════════════
# CLI INFERENCE (run directly)
# ══════════════════════════════════════════

def predict_from_path(image_path: str):
    """Standalone CLI: python app.py path/to/image.jpg"""
    path = Path(image_path)
    if not path.exists():
        print(f"❌ File not found: {image_path}")
        return

    print(f"\n📷 Loading image: {path.name}")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"❌ Could not open image: {e}")
        return

    print("🔍 Running breed classification …\n")
    result = classifier.predict(img)

    print("=" * 50)
    print(f"  🐄 Animal type : {result['animal_type']}")
    print(f"  Breed         : {result['breed']}")
    print(f"  Confidence    : {result['confidence']*100:.1f}%")
    print(f"  Inference     : {result['inference_ms']} ms")
    if result.get("demo_mode"):
        print("  ⚠️  DEMO MODE — install dependencies for real inference")
    print()
    print("  Top 5 matches:")
    for i, item in enumerate(result["top_5"], 1):
        bar = "█" * int(item["score"] * 20)
        print(f"  {i}. {item['breed']:<14} {bar:<20} {item['score']*100:.1f}%")
    print()
    m = result.get("metadata", {})
    if m:
        print(f"  Origin  : {m.get('origin','—')}")
        print(f"  Purpose : {m.get('purpose','—')}")
        print(f"  Milk    : {m.get('milk_yield','—')}")
    print("=" * 50)


# ══════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # CLI mode: python app.py image.jpg
        predict_from_path(sys.argv[1])
    else:
        # Web server mode
        print("\n🐄  Bharat Pashudhan App — AI Breed Identification Server")
        print("=" * 55)
        print(f"   Model status : {'✅ Loaded' if classifier.loaded else '⏳ Loading'}")
        print(f"   Demo mode    : {'Yes (install deps for real AI)' if classifier.demo_mode else 'No — real model active'}")
        print(f"   Breeds       : {len(BREED_METADATA)} supported")
        print("=" * 55)
        print("   API endpoints:")
        print("     GET  /api/health       — health check")
        print("     GET  /api/breeds       — breed metadata")
        print("     POST /api/predict      — classify image")
        print("     GET  /api/predict/demo — demo prediction")
        print("     GET  /                 — frontend UI")
        print("=" * 55)
        print("   Starting server on http://localhost:5000\n")
        app.run(host="0.0.0.0", port=5000, debug=False)
