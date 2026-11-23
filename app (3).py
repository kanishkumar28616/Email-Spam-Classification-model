"""
app.py  -  Flask REST API + web UI for the Bi-LSTM spam classifier.

Endpoints
    GET  /               web interface (frontend/index.html)
    GET  /health         model status  -> {"status": "ready" | "error", ...}
    POST /predict        {"text": "...", "threshold": 0.5}
                         -> {"probability", "label", "verdict", "threshold"}
    POST /predict_batch  {"messages": ["...", "..."], "threshold": 0.5}
                         -> {"results": [...], "summary": {...}}

Run from the "backend" folder (virtual environment active):
    python app.py          then open  http://127.0.0.1:5000
"""
import json
import os
import pickle
import re

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")

DEFAULT_THRESHOLD = 0.50   # probability >= threshold  ->  Spam
BORDERLINE_BAND = 0.20     # threshold-0.20 <= p < threshold  ->  Borderline
MIN_THRESHOLD, MAX_THRESHOLD = 0.05, 0.95
MAX_CHARS = 10_000         # per message
MAX_BATCH = 500            # messages per batch request

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["JSON_SORT_KEYS"] = False

state = {"model": None, "tokenizer": None, "max_len": 100, "error": None}


# ----------------------------------------------------------------------------
# Text helpers (keep identical to train_model.py)
# ----------------------------------------------------------------------------
def clean_text(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pad(sequences, maxlen: int) -> np.ndarray:
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for row, seq in enumerate(sequences):
        seq = seq[:maxlen]
        out[row, : len(seq)] = seq
    return out


# ----------------------------------------------------------------------------
# Model loading and prediction
# ----------------------------------------------------------------------------
def load_artifacts() -> None:
    """Load spam_model.h5 and tokenizer.pickle once at start-up."""
    try:
        from tensorflow.keras.models import load_model

        state["model"] = load_model(MODEL_PATH, compile=False)
        with open(TOKENIZER_PATH, "rb") as fh:
            state["tokenizer"] = pickle.load(fh)
        # the model itself tells us the sequence length it was trained with
        state["max_len"] = int(state["model"].input_shape[1])
        state["error"] = None
        print(f"Model loaded (max_len={state['max_len']})")
    except Exception as exc:  # keep the server up so /health can explain what is wrong
        state["model"] = state["tokenizer"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"
        print("Could not load model:", state["error"])


def predict_probabilities(texts):
    """Return a list of spam probabilities (0..1), one per input message."""
    cleaned = [clean_text(t) for t in texts]
    seqs = state["tokenizer"].texts_to_sequences(cleaned)
    batch = pad(seqs, state["max_len"])
    return state["model"].predict(batch, verbose=0).ravel().tolist()


def classify(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Spam"
    if probability >= max(0.0, threshold - BORDERLINE_BAND):
        return "Borderline"
    return "Safe"


# ----------------------------------------------------------------------------
# Request helpers
# ----------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def read_json() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object.")
    return data


def read_threshold(data: dict) -> float:
    try:
        value = float(data.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        raise ApiError("threshold must be a number between 0.05 and 0.95.")
    if not (MIN_THRESHOLD <= value <= MAX_THRESHOLD):
        raise ApiError(f"threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}.")
    return value


def require_model() -> None:
    if state["model"] is None or state["tokenizer"] is None:
        raise ApiError(state["error"] or "Model is not loaded.", 503)


def check_text(value, label="text") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"{label} must be a non-empty string.")
    if len(value) > MAX_CHARS:
        raise ApiError(f"{label} is longer than {MAX_CHARS} characters.")
    return value


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"error": err.message}), err.status


@app.errorhandler(404)
def handle_404(_):
    return jsonify({"error": "Not found."}), 404


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    ready = state["model"] is not None and state["tokenizer"] is not None
    return jsonify(
        {
            "status": "ready" if ready else "error",
            "max_len": state["max_len"],
            "error": state["error"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = read_json()
    text = check_text(data.get("text"))
    threshold = read_threshold(data)
    require_model()

    probability = predict_probabilities([text])[0]
    return jsonify(
        {
            "probability": round(probability, 4),
            "label": "spam" if probability >= threshold else "ham",
            "verdict": classify(probability, threshold),
            "threshold": threshold,
        }
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = read_json()
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError("messages must be a non-empty list of strings.")
    if len(messages) > MAX_BATCH:
        raise ApiError(f"A batch can hold at most {MAX_BATCH} messages.")
    messages = [check_text(m, f"messages[{i}]") for i, m in enumerate(messages)]
    threshold = read_threshold(data)
    require_model()

    probabilities = predict_probabilities(messages)
    results, counts = [], {"Spam": 0, "Borderline": 0, "Safe": 0}
    for i, (text, p) in enumerate(zip(messages, probabilities), start=1):
        verdict = classify(p, threshold)
        counts[verdict] += 1
        results.append(
            {
                "index": i,
                "text": text,
                "probability": round(p, 4),
                "label": "spam" if p >= threshold else "ham",
                "verdict": verdict,
            }
        )
    summary = {
        "total": len(results),
        "spam": counts["Spam"],
        "borderline": counts["Borderline"],
        "safe": counts["Safe"],
    }
    return jsonify({"results": results, "summary": summary, "threshold": threshold})


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    load_artifacts()
    app.run(host="127.0.0.1", port=5000, debug=False)
