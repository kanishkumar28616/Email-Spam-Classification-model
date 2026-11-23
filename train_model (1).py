"""
train_model.py  -  Train the Bi-LSTM email / SMS spam classifier.

Pipeline
    data/spam.csv  ->  clean text  ->  Keras Tokenizer  ->  pad to fixed length
    ->  Embedding + Bidirectional LSTM + Dense(sigmoid)  ->  evaluate on held-out data
    ->  save  model/model/spam_model.h5  +  model/model/tokenizer.pickle  (+ config.json)

Run from the "model" folder (with the virtual environment active):
    python train_model.py

IMPORTANT: clean_text() and pad() below are duplicated in backend/app.py.
Keep both copies identical, otherwise the API will preprocess text differently
from the way the model was trained.
"""
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError:  # newer Keras layouts
    from keras.preprocessing.text import Tokenizer

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

VOCAB_SIZE = 10000      # most frequent words kept by the tokenizer
MAX_LEN = 100           # every message is padded / truncated to this many tokens
EMBED_DIM = 64
LSTM_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.20
SEED = 42


# ----------------------------------------------------------------------------
# Text helpers (keep identical to backend/app.py)
# ----------------------------------------------------------------------------
def clean_text(text) -> str:
    """Lower-case, strip punctuation / symbols, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pad(sequences, maxlen: int = MAX_LEN) -> np.ndarray:
    """Pad (post) or truncate (post) lists of token ids to a fixed length."""
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for row, seq in enumerate(sequences):
        seq = seq[:maxlen]
        out[row, : len(seq)] = seq
    return out


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load spam.csv and return a DataFrame with columns: text, label (1 = spam, 0 = ham).

    Supports the common layouts:  v1/v2 (Kaggle SMS Spam Collection),
    label/text,  Category/Message,  or simply the first two columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {os.path.abspath(path)}")

    df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower().strip(): c for c in df.columns}

    for label_key, text_key in (("v1", "v2"), ("label", "text"), ("category", "message"), ("label", "message")):
        if label_key in cols and text_key in cols:
            label_col, text_col = cols[label_key], cols[text_key]
            break
    else:
        label_col, text_col = df.columns[0], df.columns[1]

    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0}
    )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Loaded {before} rows, {len(df)} after removing duplicate messages")
    print(df["label"].map({0: "ham", 1: "spam"}).value_counts().to_string())
    return df


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
            Bidirectional(LSTM(LSTM_UNITS)),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. data -----------------------------------------------------------------
    df = load_dataset()
    df["clean"] = df["text"].apply(clean_text)

    x_train_txt, x_test_txt, y_train, y_test = train_test_split(
        df["clean"].tolist(),
        df["label"].values,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"].values,
    )

    # 2. tokenise + pad (tokenizer is fitted on the training split only) -------
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_txt)
    x_train = pad(tokenizer.texts_to_sequences(x_train_txt))
    x_test = pad(tokenizer.texts_to_sequences(x_test_txt))

    # 3. class weights (spam is the minority class) ---------------------------
    n = len(y_train)
    n_spam = int(y_train.sum())
    n_ham = n - n_spam
    class_weight = {0: n / (2.0 * n_ham), 1: n / (2.0 * n_spam)}

    # 4. train ---------------------------------------------------------------
    model = build_model()
    model.build(input_shape=(None, MAX_LEN))
    model.summary()
    model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=2,
    )

    # 5. evaluate on held-out data ------------------------------------------
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)
    print(f"\nTest accuracy: {acc * 100:.2f}%   (loss {loss:.4f})")
    print("\nConfusion matrix [rows = actual ham/spam, cols = predicted ham/spam]")
    print(confusion_matrix(y_test, pred))
    print("\n" + classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))

    # 6. save artefacts -------------------------------------------------------
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as fh:
        pickle.dump(tokenizer, fh)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN, "test_accuracy": round(float(acc), 4)},
            fh,
            indent=2,
        )
    print(f"\nSaved:\n  {MODEL_PATH}\n  {TOKENIZER_PATH}\n  {CONFIG_PATH}")


if __name__ == "__main__":
    main()
