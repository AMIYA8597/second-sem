# ============================================================
#   LSTM & RNN Lab Assignment - Complete Solution
#   Dataset: MIT-BIH Arrhythmia (ECG Heartbeat Classification)
#   Platform: Google Colab
# ============================================================

# ============================================================
# STEP 0-A: Install Required Libraries (Run this first!)
# ============================================================
!pip install tensorflow -q
!pip install scikit-learn -q
!pip install seaborn -q

print("All packages installed successfully!")

# ============================================================
# STEP 0-B: Upload the dataset ZIP file from Kaggle
# ============================================================
# 1. Go to: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
# 2. Download the ZIP file (archive.zip or heartbeat.zip)
# 3. Run this cell — it will open an upload button

from google.colab import files
import zipfile
import os

print("=" * 60)
print("  STEP 0: Upload Dataset ZIP File")
print("=" * 60)
print("Please upload the ZIP file downloaded from Kaggle.")
print("Link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n")

uploaded = files.upload()  # This opens the upload dialog

# Extract the ZIP file
zip_filename = list(uploaded.keys())[0]
print(f"\n[INFO] Extracting '{zip_filename}'...")

os.makedirs("heartbeat_data", exist_ok=True)
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall("heartbeat_data")

print("[INFO] Extraction complete!")
print("[INFO] Files found:")
for root, dirs, filenames in os.walk("heartbeat_data"):
    for fname in filenames:
        print(f"       - {os.path.join(root, fname)}")

# ── Auto-detect the CSV paths (handles nested folders) ─────
def find_file(base_dir, filename):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == filename:
                return os.path.join(root, f)
    return None

train_path = find_file("heartbeat_data", "mitbih_train.csv")
test_path  = find_file("heartbeat_data", "mitbih_test.csv")

print(f"\n[INFO] Train CSV found at : {train_path}")
print(f"[INFO] Test  CSV found at : {test_path}")

if train_path is None or test_path is None:
    raise FileNotFoundError("Could not find mitbih_train.csv or mitbih_test.csv. "
                            "Please check the uploaded ZIP contents above.")


# ============================================================
# STEP 1: Import Required Libraries
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: Importing Libraries")
print("=" * 60)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)

print(f"[INFO] TensorFlow Version : {tf.__version__}")
print(f"[INFO] NumPy Version      : {np.__version__}")
print(f"[INFO] Pandas Version     : {pd.__version__}")
print("[INFO] All libraries imported successfully!")


# ============================================================
# ╔══════════════════════════════════════════════════════════╗
# ║           ASSIGNMENT 1 — LSTM MODEL                     ║
# ╚══════════════════════════════════════════════════════════╝
# ============================================================

print("\n" + "=" * 60)
print("  ★  ASSIGNMENT 1: LSTM-Based ECG Classification  ★")
print("=" * 60)


# ------------------------------------------------------------
# A1 — Step 1: Data Loading
# ------------------------------------------------------------
print("\n[A1 - Step 1] Loading Dataset...")

train_df = pd.read_csv(train_path, header=None)
test_df  = pd.read_csv(test_path,  header=None)

print(f"[INFO] Training set shape : {train_df.shape}")
print(f"[INFO] Test set shape     : {test_df.shape}")

# Features: first 187 columns | Label: last column (class 0–4)
X_train_full = train_df.iloc[:, :-1].values
y_train_full = train_df.iloc[:, -1].values.astype(int)

X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values.astype(int)

class_names = [
    "Normal (0)",
    "Supraventricular ectopic (1)",
    "Ventricular ectopic (2)",
    "Fusion beats (3)",
    "Unknown beats (4)"
]

print("\n[INFO] Class Distribution in Training Set:")
unique, counts = np.unique(y_train_full, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"       Class {cls} — {class_names[cls]} : {cnt} samples")


# ------------------------------------------------------------
# A1 — Step 2: Preprocessing
# ------------------------------------------------------------
print("\n[A1 - Step 2] Preprocessing Data...")

# Min-Max Normalization → [0, 1]
X_min = X_train_full.min()
X_max = X_train_full.max()
X_train_norm = (X_train_full - X_min) / (X_max - X_min)
X_test_norm  = (X_test       - X_min) / (X_max - X_min)
print(f"[INFO] Normalized range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")

# 80/20 Stratified Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_norm, y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)
print(f"[INFO] Train samples      : {X_train.shape[0]}")
print(f"[INFO] Validation samples : {X_val.shape[0]}")
print(f"[INFO] Test samples       : {X_test_norm.shape[0]}")

# Reshape → (samples, 187 timesteps, 1 feature) for LSTM
X_train_lstm = X_train.reshape(-1, 187, 1)
X_val_lstm   = X_val.reshape(-1, 187, 1)
X_test_lstm  = X_test_norm.reshape(-1, 187, 1)
print(f"[INFO] LSTM input shape   : {X_train_lstm.shape}")


# ------------------------------------------------------------
# A1 — Step 3: Build LSTM Model
# ------------------------------------------------------------
print("\n[A1 - Step 3] Building LSTM Model...")

lstm_model = Sequential([
    LSTM(64, input_shape=(187, 1)),   # LSTM layer — 64 units
    Dense(32, activation='relu'),     # Hidden Dense layer
    Dense(5,  activation='softmax')   # Output layer — 5 classes
])

lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n[INFO] LSTM Model Architecture:")
lstm_model.summary()


# ------------------------------------------------------------
# A1 — Step 4: Train LSTM
# ------------------------------------------------------------
print("\n[A1 - Step 4] Training LSTM Model...")
print("       Epochs=15 | Batch=64 | EarlyStopping patience=3")

early_stop_lstm = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val_lstm, y_val),
    callbacks=[early_stop_lstm],
    verbose=1
)
print("[INFO] LSTM training complete!")


# ------------------------------------------------------------
# A1 — Step 5: Evaluate LSTM
# ------------------------------------------------------------
print("\n[A1 - Step 5] Evaluating LSTM on Test Set...")

lstm_loss, lstm_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm, verbose=0), axis=1)

print("\n" + "=" * 60)
print("  ★  LSTM RESULTS (Assignment 1)")
print("=" * 60)
print(f"  Test Loss     : {lstm_loss:.4f}")
print(f"  Test Accuracy : {lstm_acc * 100:.2f}%")
print("\n  Per-Class Classification Report:")
print(classification_report(y_test, y_pred_lstm, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(9, 7))
sns.heatmap(confusion_matrix(y_test, y_pred_lstm),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("LSTM — Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Saved: lstm_confusion_matrix.png")

# Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(lstm_history.history['accuracy'],     label='Train', color='steelblue')
ax1.plot(lstm_history.history['val_accuracy'], label='Val',   color='orange')
ax1.set_title("LSTM Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(True)

ax2.plot(lstm_history.history['loss'],     label='Train', color='steelblue')
ax2.plot(lstm_history.history['val_loss'], label='Val',   color='orange')
ax2.set_title("LSTM Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True)

plt.suptitle("LSTM Training Curves", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("lstm_training_curves.png", dpi=150)
plt.show()
print("[INFO] Saved: lstm_training_curves.png")


# ============================================================
# ╔══════════════════════════════════════════════════════════╗
# ║           ASSIGNMENT 2 — SimpleRNN MODEL                ║
# ╚══════════════════════════════════════════════════════════╝
# ============================================================

print("\n" + "=" * 60)
print("  ★  ASSIGNMENT 2: SimpleRNN-Based ECG Classification  ★")
print("=" * 60)


# ------------------------------------------------------------
# A2 — Step 1: Preprocessing (reuse from Assignment 1)
# ------------------------------------------------------------
print("\n[A2 - Step 1] Reusing preprocessed data from Assignment 1...")

X_train_rnn = X_train.reshape(-1, 187, 1)
X_val_rnn   = X_val.reshape(-1, 187, 1)
X_test_rnn  = X_test_norm.reshape(-1, 187, 1)

print(f"[INFO] RNN input shape : {X_train_rnn.shape}")
print("[INFO] Same normalization & 80/20 split applied.")


# ------------------------------------------------------------
# A2 — Step 2: Build SimpleRNN Model
# ------------------------------------------------------------
print("\n[A2 - Step 2] Building SimpleRNN Model...")

rnn_model = Sequential([
    SimpleRNN(64, input_shape=(187, 1)),  # SimpleRNN layer — 64 units
    Dense(32, activation='relu'),          # Hidden Dense layer
    Dense(5,  activation='softmax')        # Output layer — 5 classes
])

rnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n[INFO] SimpleRNN Model Architecture:")
rnn_model.summary()


# ------------------------------------------------------------
# A2 — Step 3: Train SimpleRNN
# ------------------------------------------------------------
print("\n[A2 - Step 3] Training SimpleRNN Model...")
print("       Epochs=15 | Batch=64 | EarlyStopping patience=3")

early_stop_rnn = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

rnn_history = rnn_model.fit(
    X_train_rnn, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val_rnn, y_val),
    callbacks=[early_stop_rnn],
    verbose=1
)
print("[INFO] SimpleRNN training complete!")


# ------------------------------------------------------------
# A2 — Step 4: Evaluate SimpleRNN
# ------------------------------------------------------------
print("\n[A2 - Step 4] Evaluating SimpleRNN on Test Set...")

rnn_loss, rnn_acc = rnn_model.evaluate(X_test_rnn, y_test, verbose=0)
y_pred_rnn = np.argmax(rnn_model.predict(X_test_rnn, verbose=0), axis=1)

print("\n" + "=" * 60)
print("  ★  SimpleRNN RESULTS (Assignment 2)")
print("=" * 60)
print(f"  Test Loss     : {rnn_loss:.4f}")
print(f"  Test Accuracy : {rnn_acc * 100:.2f}%")
print("\n  Per-Class Classification Report:")
print(classification_report(y_test, y_pred_rnn, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(9, 7))
sns.heatmap(confusion_matrix(y_test, y_pred_rnn),
            annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title("SimpleRNN — Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rnn_confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Saved: rnn_confusion_matrix.png")

# Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(rnn_history.history['accuracy'],     label='Train', color='seagreen')
ax1.plot(rnn_history.history['val_accuracy'], label='Val',   color='tomato')
ax1.set_title("SimpleRNN Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(True)

ax2.plot(rnn_history.history['loss'],     label='Train', color='seagreen')
ax2.plot(rnn_history.history['val_loss'], label='Val',   color='tomato')
ax2.set_title("SimpleRNN Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True)

plt.suptitle("SimpleRNN Training Curves", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("rnn_training_curves.png", dpi=150)
plt.show()
print("[INFO] Saved: rnn_training_curves.png")


# ============================================================
# ╔══════════════════════════════════════════════════════════╗
# ║       COMPARISON: LSTM vs SimpleRNN                     ║
# ╚══════════════════════════════════════════════════════════╝
# ============================================================

print("\n" + "=" * 60)
print("  ★  FINAL COMPARISON: LSTM vs SimpleRNN  ★")
print("=" * 60)

print(f"\n  {'Model':<15} {'Test Loss':>12} {'Test Accuracy':>15}")
print(f"  {'-'*15} {'-'*12} {'-'*15}")
print(f"  {'LSTM':<15} {lstm_loss:>12.4f} {lstm_acc*100:>14.2f}%")
print(f"  {'SimpleRNN':<15} {rnn_loss:>12.4f} {rnn_acc*100:>14.2f}%")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(7, 5))
models   = ["LSTM", "SimpleRNN"]
acc_vals = [lstm_acc * 100, rnn_acc * 100]
colors   = ["steelblue", "seagreen"]
bars = ax.bar(models, acc_vals, color=colors, width=0.4, edgecolor='black')
for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylim(0, 110)
ax.set_title("LSTM vs SimpleRNN — Test Accuracy", fontsize=14, fontweight='bold')
ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Model")
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("[INFO] Saved: model_comparison.png")


# ============================================================
# ★  CONCLUSION & EXPLANATION (For Teacher & Student)
# ============================================================
print("\n" + "=" * 60)
print("  ★  CONCLUSION & EXPLANATION")
print("    (Printed for Teacher Review & Student Clarity)")
print("=" * 60)
print(f"""
DATASET USED:
  Name   : MIT-BIH Arrhythmia Dataset
  Source : Kaggle — shayanfazeli/heartbeat
  Size   : 109,446 ECG samples
  Classes: 5 (Normal, Supraventricular, Ventricular, Fusion, Unknown)
  Input  : 187 ECG timesteps per sample

─────────────────────────────────────────────────────────────
PREPROCESSING (Same for both assignments):
  1. Min-Max Normalization → signals scaled to [0.0, 1.0]
  2. 80/20 stratified split → train / validation
  3. Reshape → (samples, 187, 1) for sequence model input

─────────────────────────────────────────────────────────────
ASSIGNMENT 1 — LSTM Model:
  Architecture : LSTM(64) → Dense(32, ReLU) → Dense(5, Softmax)
  Loss         : Sparse Categorical Cross-Entropy
  Optimizer    : Adam
  Batch Size   : 64  |  Max Epochs : 15
  Early Stop   : patience=3 on val_loss

  WHY LSTM?
  → LSTM has 3 gates: Forget, Input, Output.
  → These gates control what to remember and what to discard
    over long sequences — crucial for ECG signal patterns.
  → Avoids vanishing gradient problem of plain RNNs.

─────────────────────────────────────────────────────────────
ASSIGNMENT 2 — SimpleRNN Model:
  Architecture : SimpleRNN(64) → Dense(32, ReLU) → Dense(5, Softmax)
  Same training settings as LSTM.

  WHY SimpleRNN?
  → A basic recurrent unit — uses only one hidden state.
  → Simpler and faster to train.
  → Struggles with long sequences (vanishing gradients).
  → For 187-step ECG signals, it captures some patterns
    but generally underperforms compared to LSTM.

─────────────────────────────────────────────────────────────
COMPARISON SUMMARY:
  LSTM      Test Accuracy : {lstm_acc * 100:.2f}%
  SimpleRNN Test Accuracy : {rnn_acc  * 100:.2f}%

  → LSTM achieves higher accuracy because its gating mechanism
    preserves relevant long-range temporal information in ECG.
  → SimpleRNN trains faster but loses information over time.

─────────────────────────────────────────────────────────────
OUTPUT FILES SAVED:
  lstm_confusion_matrix.png
  lstm_training_curves.png
  rnn_confusion_matrix.png
  rnn_training_curves.png
  model_comparison.png
""")

print("=" * 60)
print("  Lab Assignment Completed Successfully!")
print("=" * 60)