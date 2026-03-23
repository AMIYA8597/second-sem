# ============================================================
# FULL GRU SENTIMENT ANALYSIS - SINGLE CELL
# ============================================================

# ── STEP 1: Install & Imports ────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "nltk", "-q"])

import numpy as np
import pandas as pd
import nltk
import re
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

nltk.download('stopwords', quiet=True)
print("✅ STEP 1 DONE — TensorFlow:", tf.__version__)


# ── STEP 2: Load Dataset ─────────────────────────────────────
try:
    df = pd.read_csv('/content/IMDB Dataset.csv',
                     encoding='utf-8', engine='python', on_bad_lines='skip')
    print("✅ STEP 2 DONE — Loaded with utf-8")
except Exception as e:
    print(f"utf-8 failed → trying latin-1...")
    df = pd.read_csv('/content/IMDB Dataset.csv', encoding='latin-1')
    print("✅ STEP 2 DONE — Loaded with latin-1")

print("Shape:", df.shape)
print(df['sentiment'].value_counts())


# ── STEP 3: Clean & Preprocess Text ──────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['label']   = df['sentiment'].map({'positive': 1, 'negative': 0})
df['cleaned'] = df['review'].apply(clean_text)

print("✅ STEP 3 DONE — Text cleaned")
print("Sample:", df['cleaned'][0][:120])


# ── STEP 4: Tokenize & Pad ───────────────────────────────────
VOCAB_SIZE  = 10000
MAX_LENGTH  = 200
EMBED_DIM   = 64

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
padded    = pad_sequences(sequences, maxlen=MAX_LENGTH,
                          padding='post', truncating='post')
labels    = np.array(df['label'])

print("✅ STEP 4 DONE — Padded shape:", padded.shape)


# ── STEP 5: Train / Test Split (80-20) ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("✅ STEP 5 DONE")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)


# ── STEP 6: Build GRU Model ──────────────────────────────────
tf.random.set_seed(42)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM,
              input_length=MAX_LENGTH),
    GRU(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
print("✅ STEP 6 DONE — Model built")


# ── STEP 7: Train the Model ──────────────────────────────────
EPOCHS     = 5
BATCH_SIZE = 64

print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)
print("✅ STEP 7 DONE — Training complete")


# ── STEP 8: Evaluate on Test Set ─────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = (y_pred_prob >= 0.5).astype(int).flatten()

print("\n" + "=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Negative', 'Positive']))
print("✅ STEP 8 DONE — Evaluation complete")


# ── STEP 9: Plot Training History ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(history.history['accuracy'],
             label='Train Accuracy', marker='o', color='blue')
axes[0].plot(history.history['val_accuracy'],
             label='Val Accuracy',   marker='o', color='orange')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],
             label='Train Loss', marker='o', color='blue')
axes[1].plot(history.history['val_loss'],
             label='Val Loss',   marker='o', color='orange')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("✅ STEP 9 DONE — Plot saved")


# ── STEP 10: Test on Custom Reviews ──────────────────────────
def predict_sentiment(review_text):
    cleaned   = clean_text(review_text)
    seq       = tokenizer.texts_to_sequences([cleaned])
    padded_in = pad_sequences(seq, maxlen=MAX_LENGTH,
                               padding='post', truncating='post')
    prob  = model.predict(padded_in, verbose=0)[0][0]
    label = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
    print(f"\nReview   : {review_text[:80]}")
    print(f"Sentiment: {label}  (confidence: {prob:.4f})")

print("\n── Custom Review Tests ──────────────────────────────")
predict_sentiment("This movie was absolutely fantastic! I loved every moment of it.")
predict_sentiment("Terrible film. Waste of time, extremely boring and poorly acted.")
predict_sentiment("It was okay, not great but not bad either.")

print("\n✅ ALL STEPS COMPLETE!")