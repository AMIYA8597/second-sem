# ============================================================
# ASSIGNMENT: Manual Multi-Head Attention from Scratch
# Input: "I love AI"  |  n=3, d_model=4, h=2 heads
# ============================================================

import numpy as np

np.random.seed(42)   # for reproducibility

# ── STEP 1: Given Embedding Matrix X (3×4) ──────────────────
# "I"=row0, "love"=row1, "AI"=row2
X = np.array([
    [1, 0, 1, 0],   # "I"
    [0, 1, 0, 1],   # "love"
    [1, 1, 1, 1],   # "AI"
], dtype=float)

print("=" * 55)
print("INPUT EMBEDDING MATRIX X  (shape:", X.shape, ")")
print("=" * 55)
print(X)

# ── STEP 2: Parameters ──────────────────────────────────────
n       = 3   # sequence length
d_model = 4   # embedding dimension
h       = 2   # number of heads
d_k     = d_model // h   # dimension per head = 2
d_v     = d_model // h   # dimension per head = 2

print(f"\nd_model={d_model}, h={h}, d_k={d_k}, d_v={d_v}")

# ── STEP 3: Random Weight Matrices ──────────────────────────
# Each head has its own Wq, Wk, Wv projection (d_model × d_k)
# Plus one final W_O projection (h*d_v × d_model)

print("\n" + "=" * 55)
print("WEIGHT MATRICES (randomly chosen)")
print("=" * 55)

# Head 1 weights
Wq1 = np.random.randn(d_model, d_k)
Wk1 = np.random.randn(d_model, d_k)
Wv1 = np.random.randn(d_model, d_v)

# Head 2 weights
Wq2 = np.random.randn(d_model, d_k)
Wk2 = np.random.randn(d_model, d_k)
Wv2 = np.random.randn(d_model, d_v)

# Final output projection (h*d_v × d_model) = (4 × 4)
W_O = np.random.randn(h * d_v, d_model)

print("\nWq1:\n", np.round(Wq1, 3))
print("Wk1:\n", np.round(Wk1, 3))
print("Wv1:\n", np.round(Wv1, 3))
print("\nWq2:\n", np.round(Wq2, 3))
print("Wk2:\n", np.round(Wk2, 3))
print("Wv2:\n", np.round(Wv2, 3))
print("\nW_O:\n", np.round(W_O, 3))

# ── STEP 4: Softmax Function ─────────────────────────────────
def softmax(x):
    """Row-wise softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # stable softmax
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ── STEP 5: Scaled Dot-Product Attention Function ───────────
def scaled_dot_product_attention(Q, K, V, d_k, head_name=""):
    """
    Attention(Q,K,V) = Softmax(QK^T / sqrt(d_k)) * V
    """
    print(f"\n  --- {head_name} ---")

    # Step A: Q, K, V
    print(f"  Q:\n{np.round(Q, 4)}")
    print(f"  K:\n{np.round(K, 4)}")
    print(f"  V:\n{np.round(V, 4)}")

    # Step B: Attention Scores = Q @ K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)
    print(f"\n  Attention Scores  (QK^T / sqrt({d_k})):\n{np.round(scores, 4)}")

    # Step C: Softmax
    weights = softmax(scores)
    print(f"\n  Softmax (Attention Weights):\n{np.round(weights, 4)}")

    # Step D: Weighted sum of V
    output = weights @ V
    print(f"\n  Head Output  (Attention Weights @ V):\n{np.round(output, 4)}")

    return output, weights, scores

# ── STEP 6: Compute Q, K, V for Each Head ────────────────────
print("\n" + "=" * 55)
print("MULTI-HEAD ATTENTION — STEP BY STEP")
print("=" * 55)

# HEAD 1
Q1 = X @ Wq1   # (3,4)@(4,2) = (3,2)
K1 = X @ Wk1
V1 = X @ Wv1

# HEAD 2
Q2 = X @ Wq2
K2 = X @ Wk2
V2 = X @ Wv2

print("\n[HEAD 1] Q1, K1, V1:")
head1_out, head1_weights, head1_scores = scaled_dot_product_attention(Q1, K1, V1, d_k, "HEAD 1")

print("\n[HEAD 2] Q2, K2, V2:")
head2_out, head2_weights, head2_scores = scaled_dot_product_attention(Q2, K2, V2, d_k, "HEAD 2")

# ── STEP 7: Concatenate Head Outputs ─────────────────────────
Z_concat = np.concatenate([head1_out, head2_out], axis=-1)  # (3,2) + (3,2) → (3,4)

print("\n" + "=" * 55)
print("CONCATENATED HEAD OUTPUTS  Z_concat  (shape:", Z_concat.shape, ")")
print("=" * 55)
print(np.round(Z_concat, 4))

# ── STEP 8: Final Linear Projection ─────────────────────────
# Output = Z_concat @ W_O
final_output = Z_concat @ W_O   # (3,4)@(4,4) = (3,4)

print("\n" + "=" * 55)
print("FINAL OUTPUT = Z_concat @ W_O  (shape:", final_output.shape, ")")
print("=" * 55)
print(np.round(final_output, 4))

# ── STEP 9: Summary Table ────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY OF ALL INTERMEDIATE MATRICES")
print("=" * 55)
print(f"\n{'Matrix':<25} {'Shape':<15} Description")
print("-" * 65)
print(f"{'X':<25} {str(X.shape):<15} Input embedding matrix")
print(f"{'Q1 = X @ Wq1':<25} {str(Q1.shape):<15} Query matrix (Head 1)")
print(f"{'K1 = X @ Wk1':<25} {str(K1.shape):<15} Key matrix (Head 1)")
print(f"{'V1 = X @ Wv1':<25} {str(V1.shape):<15} Value matrix (Head 1)")
print(f"{'Q2 = X @ Wq2':<25} {str(Q2.shape):<15} Query matrix (Head 2)")
print(f"{'K2 = X @ Wk2':<25} {str(K2.shape):<15} Key matrix (Head 2)")
print(f"{'V2 = X @ Wv2':<25} {str(V2.shape):<15} Value matrix (Head 2)")
print(f"{'Scores (Head 1)':<25} {str(head1_scores.shape):<15} QK^T/sqrt(d_k)  Head 1")
print(f"{'Softmax (Head 1)':<25} {str(head1_weights.shape):<15} Attention weights Head 1")
print(f"{'Head1 Output':<25} {str(head1_out.shape):<15} Weighted V Head 1")
print(f"{'Scores (Head 2)':<25} {str(head2_scores.shape):<15} QK^T/sqrt(d_k)  Head 2")
print(f"{'Softmax (Head 2)':<25} {str(head2_weights.shape):<15} Attention weights Head 2")
print(f"{'Head2 Output':<25} {str(head2_out.shape):<15} Weighted V Head 2")
print(f"{'Z_concat':<25} {str(Z_concat.shape):<15} Concat(Head1, Head2)")
print(f"{'Final Output':<25} {str(final_output.shape):<15} Z_concat @ W_O")

# ── STEP 10: Visualize Attention Weights ─────────────────────
import matplotlib.pyplot as plt

tokens = ["I", "love", "AI"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for i, (weights, title) in enumerate(zip(
    [head1_weights, head2_weights],
    ["Head 1 Attention Weights", "Head 2 Attention Weights"]
)):
    im = axes[i].imshow(weights, cmap="Blues", vmin=0, vmax=1)
    axes[i].set_xticks(range(n))
    axes[i].set_yticks(range(n))
    axes[i].set_xticklabels(tokens, fontsize=12)
    axes[i].set_yticklabels(tokens, fontsize=12)
    axes[i].set_xlabel("Key (attended to)", fontsize=11)
    axes[i].set_ylabel("Query (attends from)", fontsize=11)
    axes[i].set_title(title, fontsize=13, fontweight='bold')
    # Annotate cells
    for r in range(n):
        for c in range(n):
            axes[i].text(c, r, f"{weights[r,c]:.2f}",
                         ha='center', va='center', fontsize=11,
                         color='white' if weights[r,c] > 0.6 else 'black')
    plt.colorbar(im, ax=axes[i])

plt.suptitle('Multi-Head Self-Attention — "I love AI"', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()




















































































































































































































































































































































# %matplotlib inline      

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# print("TensorFlow:", tf.__version__)


# # ══════════════════════════════════════════════════════════════════
# # ████  ASSIGNMENT 1 : TRANSFORMER SENTIMENT CLASSIFIER  ████
# # ══════════════════════════════════════════════════════════════════

# # ── CELL 2 : Dataset ─────────────────────────────────────────────
# texts = [
#     "i love this movie", "this is amazing", "i like it", "so good",
#     "i hate this",       "this is bad",     "terrible experience", "not good"
# ] * 20                               # ~160 samples

# labels = [1, 1, 1, 1, 0, 0, 0, 0] * 20

# print(f"Total samples : {len(texts)}")
# print(f"Positive examples : {sum(labels)}  |  Negative : {len(labels)-sum(labels)}")


# # ── CELL 3 : Tokenisation ────────────────────────────────────────
# vocab_size = 1000
# max_len    = 10

# vectorizer = layers.TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=max_len
# )
# vectorizer.adapt(texts)

# X = vectorizer(np.array(texts))
# y = np.array(labels)

# print(f"X shape : {X.shape}   y shape : {y.shape}")
# print(f"Sample tokenised row  : {X[0].numpy()}")


# # ── CELL 4 : Positional Embedding Layer ──────────────────────────
# class PositionalEmbedding(layers.Layer):
#     def __init__(self, max_len, vocab_size, embed_dim):
#         super().__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb   = layers.Embedding(input_dim=max_len,    output_dim=embed_dim)

#     def call(self, x):
#         positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
#         return self.token_emb(x) + self.pos_emb(positions)


# # ── CELL 5 : Transformer Block ───────────────────────────────────
# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim):
#         super().__init__()
#         self.att   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn   = keras.Sequential([
#             layers.Dense(ff_dim,     activation="relu"),
#             layers.Dense(embed_dim),
#         ])
#         self.norm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.norm2 = layers.LayerNormalization(epsilon=1e-6)

#     def call(self, inputs):
#         attn_out = self.att(inputs, inputs)          # Self-Attention
#         out1     = self.norm1(inputs + attn_out)     # Add & Norm
#         ffn_out  = self.ffn(out1)                    # Feed-Forward
#         return self.norm2(out1 + ffn_out)            # Add & Norm


# # ── CELL 6 : Build Model ─────────────────────────────────────────
# embed_dim = 32
# num_heads = 2
# ff_dim    = 32

# inputs = layers.Input(shape=(max_len,))
# x = PositionalEmbedding(max_len, vocab_size, embed_dim)(inputs)  # Positional Embedding
# x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)             # Transformer Block
# x = layers.GlobalAveragePooling1D()(x)                            # Global Avg Pool
# x = layers.Dense(32, activation="relu")(x)                        # Dense relu
# outputs = layers.Dense(1, activation="sigmoid")(x)                # Dense sigmoid

# model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()


# # ── CELL 7 : Compile & Train ─────────────────────────────────────
# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# history = model.fit(
#     X, y,
#     epochs=10,
#     batch_size=8,
#     validation_split=0.2,
#     verbose=1
# )


# # ── CELL 8 : Plot Accuracy & Loss Curves ─────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# epochs_range = range(1, 11)

# # ── Accuracy ──
# axes[0].plot(epochs_range, history.history["accuracy"],
#              marker='o', linewidth=2, color='royalblue', label="Train Accuracy")
# axes[0].plot(epochs_range, history.history["val_accuracy"],
#              marker='s', linewidth=2, color='tomato',    label="Val Accuracy",  linestyle='--')
# axes[0].set_title("Accuracy Curve", fontsize=14, fontweight='bold')
# axes[0].set_xlabel("Epoch",    fontsize=12)
# axes[0].set_ylabel("Accuracy", fontsize=12)
# axes[0].set_xticks(epochs_range)
# axes[0].legend(fontsize=11)
# axes[0].grid(True, alpha=0.4)
# # Annotate final values
# axes[0].annotate(f"{history.history['accuracy'][-1]:.3f}",
#                  xy=(10, history.history['accuracy'][-1]),
#                  xytext=(8.5, history.history['accuracy'][-1]-0.05),
#                  fontsize=10, color='royalblue')
# axes[0].annotate(f"{history.history['val_accuracy'][-1]:.3f}",
#                  xy=(10, history.history['val_accuracy'][-1]),
#                  xytext=(8.5, history.history['val_accuracy'][-1]+0.02),
#                  fontsize=10, color='tomato')

# # ── Loss ──
# axes[1].plot(epochs_range, history.history["loss"],
#              marker='o', linewidth=2, color='darkorange', label="Train Loss")
# axes[1].plot(epochs_range, history.history["val_loss"],
#              marker='s', linewidth=2, color='purple',     label="Val Loss", linestyle='--')
# axes[1].set_title("Loss Curve", fontsize=14, fontweight='bold')
# axes[1].set_xlabel("Epoch", fontsize=12)
# axes[1].set_ylabel("Loss",  fontsize=12)
# axes[1].set_xticks(epochs_range)
# axes[1].legend(fontsize=11)
# axes[1].grid(True, alpha=0.4)
# axes[1].annotate(f"{history.history['loss'][-1]:.3f}",
#                  xy=(10, history.history['loss'][-1]),
#                  xytext=(8.5, history.history['loss'][-1]+0.02),
#                  fontsize=10, color='darkorange')
# axes[1].annotate(f"{history.history['val_loss'][-1]:.3f}",
#                  xy=(10, history.history['val_loss'][-1]),
#                  xytext=(8.5, history.history['val_loss'][-1]-0.04),
#                  fontsize=10, color='purple')

# plt.suptitle("Assignment 1 — Transformer Sentiment Classifier Training",
#              fontsize=15, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.savefig("assignment1_curves.png", dpi=150, bbox_inches='tight')
# plt.show()

# # ── Final accuracy ────────────────────────────────────────────────
# final_train_acc = history.history["accuracy"][-1]
# final_val_acc   = history.history["val_accuracy"][-1]
# final_train_loss= history.history["loss"][-1]
# final_val_loss  = history.history["val_loss"][-1]
# print(f"\n{'='*45}")
# print(f"  Final Train Accuracy : {final_train_acc:.4f}")
# print(f"  Final Val   Accuracy : {final_val_acc:.4f}")
# print(f"  Final Train Loss     : {final_train_loss:.4f}")
# print(f"  Final Val   Loss     : {final_val_loss:.4f}")
# print(f"{'='*45}")


# # ══════════════════════════════════════════════════════════════════
# # ████  ASSIGNMENT 2 : MANUAL MULTI-HEAD ATTENTION  ████
# # Input : "I love AI"   n=3, d_model=4, h=2
# # ══════════════════════════════════════════════════════════════════

# # ── CELL 9 : Setup ───────────────────────────────────────────────
# print("\n" + "═"*60)
# print("  ASSIGNMENT 2 : Manual Multi-Head Attention")
# print("  Input : 'I love AI'   |   n=3, d_model=4, h=2")
# print("═"*60)

# np.random.seed(42)

# # Given embedding matrix X  (shape 3×4)
# # Row 0 = "I",  Row 1 = "love",  Row 2 = "AI"
# X2 = np.array([
#     [1, 0, 1, 0],
#     [0, 1, 0, 1],
#     [1, 1, 1, 1],
# ], dtype=float)

# n       = 3          # sequence length
# d_model = 4          # embedding dim
# h       = 2          # number of heads
# d_k     = d_model // h   # = 2  (key/query dim per head)
# d_v     = d_model // h   # = 2  (value dim per head)

# print(f"\nGiven X (shape {X2.shape}):\n{X2}")
# print(f"\nParameters: n={n}, d_model={d_model}, h={h}, d_k={d_k}, d_v={d_v}")


# # ── CELL 10 : Weight Matrices ────────────────────────────────────
# # Each head: Wq, Wk, Wv  shape (d_model × d_k) = (4×2)
# # Final W_O               shape (h*d_v  × d_model) = (4×4)

# Wq = [np.random.randn(d_model, d_k) for _ in range(h)]
# Wk = [np.random.randn(d_model, d_k) for _ in range(h)]
# Wv = [np.random.randn(d_model, d_v) for _ in range(h)]
# W_O = np.random.randn(h * d_v, d_model)

# print("\n── Randomly chosen weight matrices ──")
# for i in range(h):
#     print(f"\nHead {i+1}:")
#     print(f"  Wq{i+1} (shape {Wq[i].shape}):\n{np.round(Wq[i], 4)}")
#     print(f"  Wk{i+1} (shape {Wk[i].shape}):\n{np.round(Wk[i], 4)}")
#     print(f"  Wv{i+1} (shape {Wv[i].shape}):\n{np.round(Wv[i], 4)}")
# print(f"\nW_O (shape {W_O.shape}):\n{np.round(W_O, 4)}")


# # ── CELL 11 : Softmax & Attention Functions ───────────────────────
# def softmax(x):
#     """Numerically stable row-wise softmax."""
#     e = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return e / e.sum(axis=-1, keepdims=True)

# def single_head_attention(Q, K, V, d_k, head_idx):
#     """
#     Scaled Dot-Product Attention.
#     Returns: output, attention_weights, raw_scores
#     """
#     print(f"\n{'─'*50}")
#     print(f"  HEAD {head_idx}  —  Intermediate Matrices")
#     print(f"{'─'*50}")

#     print(f"\n  Q{head_idx}  (shape {Q.shape})  =  X @ Wq{head_idx}:\n{np.round(Q, 4)}")
#     print(f"\n  K{head_idx}  (shape {K.shape})  =  X @ Wk{head_idx}:\n{np.round(K, 4)}")
#     print(f"\n  V{head_idx}  (shape {V.shape})  =  X @ Wv{head_idx}:\n{np.round(V, 4)}")

#     # Step 1 : Raw scores  =  Q @ Kᵀ
#     raw_scores = Q @ K.T
#     print(f"\n  Step 1 — Q{head_idx} @ K{head_idx}ᵀ  (shape {raw_scores.shape}):\n{np.round(raw_scores, 4)}")

#     # Step 2 : Scale by 1/√d_k
#     scaled_scores = raw_scores / np.sqrt(d_k)
#     print(f"\n  Step 2 — Scaled scores  ÷ √{d_k}={np.sqrt(d_k):.4f}  (shape {scaled_scores.shape}):\n{np.round(scaled_scores, 4)}")

#     # Step 3 : Softmax
#     attn_weights = softmax(scaled_scores)
#     print(f"\n  Step 3 — Softmax (attention weights)  (shape {attn_weights.shape}):\n{np.round(attn_weights, 4)}")
#     print(f"           Row sums (must be 1): {np.round(attn_weights.sum(axis=1), 6)}")

#     # Step 4 : Weighted sum of V
#     head_out = attn_weights @ V
#     print(f"\n  Step 4 — Head Output = Softmax @ V{head_idx}  (shape {head_out.shape}):\n{np.round(head_out, 4)}")

#     return head_out, attn_weights, scaled_scores


# # ── CELL 12 : Run Both Heads ─────────────────────────────────────
# print("\n" + "═"*60)
# print("  COMPUTING Q, K, V AND ATTENTION FOR EACH HEAD")
# print("═"*60)

# head_outputs  = []
# all_weights   = []
# all_scores    = []

# for i in range(h):
#     Q_i = X2 @ Wq[i]     # (3,4)@(4,2) → (3,2)
#     K_i = X2 @ Wk[i]
#     V_i = X2 @ Wv[i]

#     out, weights, scores = single_head_attention(Q_i, K_i, V_i, d_k, i+1)
#     head_outputs.append(out)
#     all_weights.append(weights)
#     all_scores.append(scores)


# # ── CELL 13 : Concatenate & Final Projection ─────────────────────
# print("\n" + "═"*60)
# print("  CONCATENATION & FINAL LINEAR PROJECTION")
# print("═"*60)

# Z_concat = np.concatenate(head_outputs, axis=-1)   # (3,2)+(3,2) → (3,4)
# print(f"\n  Z_concat = Concat(Head1, Head2)  (shape {Z_concat.shape}):\n{np.round(Z_concat, 4)}")

# final_output = Z_concat @ W_O                      # (3,4)@(4,4) → (3,4)
# print(f"\n  Final Output = Z_concat @ W_O  (shape {final_output.shape}):\n{np.round(final_output, 4)}")


# # ── CELL 14 : Full Summary ───────────────────────────────────────
# print("\n" + "═"*60)
# print("  COMPLETE SUMMARY — ALL INTERMEDIATE MATRICES")
# print("═"*60)
# rows = [
#     ("X  (given)",              X2.shape,              "Input embedding matrix"),
#     ("Q1 = X @ Wq1",           (n, d_k),               "Query — Head 1"),
#     ("K1 = X @ Wk1",           (n, d_k),               "Key — Head 1"),
#     ("V1 = X @ Wv1",           (n, d_v),               "Value — Head 1"),
#     ("Scores1 = Q1@K1ᵀ/√d_k", (n, n),                  "Attention scores — Head 1"),
#     ("Softmax1",               (n, n),                  "Attention weights — Head 1"),
#     ("Head1 Output",           (n, d_v),               "Weighted V — Head 1"),
#     ("Q2 = X @ Wq2",           (n, d_k),               "Query — Head 2"),
#     ("K2 = X @ Wk2",           (n, d_k),               "Key — Head 2"),
#     ("V2 = X @ Wv2",           (n, d_v),               "Value — Head 2"),
#     ("Scores2 = Q2@K2ᵀ/√d_k", (n, n),                  "Attention scores — Head 2"),
#     ("Softmax2",               (n, n),                  "Attention weights — Head 2"),
#     ("Head2 Output",           (n, d_v),               "Weighted V — Head 2"),
#     ("Z_concat",               (n, h*d_v),             "Concat(Head1, Head2)"),
#     ("Final Output",           (n, d_model),           "Z_concat @ W_O  ← FINAL"),
# ]
# print(f"\n  {'Matrix':<28} {'Shape':<12} Description")
# print("  " + "-"*65)
# for name, shape, desc in rows:
#     print(f"  {name:<28} {str(shape):<12} {desc}")


# # ── CELL 15 : Attention Heatmaps ─────────────────────────────────
# tokens = ["I", "love", "AI"]

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# for i, (weights, title) in enumerate(zip(
#     all_weights,
#     ["Head 1 — Attention Weights", "Head 2 — Attention Weights"]
# )):
#     im = axes[i].imshow(weights, cmap="YlOrRd", vmin=0, vmax=1)
#     axes[i].set_xticks(range(n))
#     axes[i].set_yticks(range(n))
#     axes[i].set_xticklabels(tokens, fontsize=13)
#     axes[i].set_yticklabels(tokens, fontsize=13)
#     axes[i].set_xlabel("Key  (attended to)",    fontsize=12)
#     axes[i].set_ylabel("Query  (attends from)", fontsize=12)
#     axes[i].set_title(title, fontsize=13, fontweight='bold', pad=12)
#     for r in range(n):
#         for c in range(n):
#             v = weights[r, c]
#             axes[i].text(c, r, f"{v:.3f}",
#                          ha='center', va='center', fontsize=12,
#                          color='white' if v > 0.55 else 'black',
#                          fontweight='bold')
#     plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

# plt.suptitle('Assignment 2 — Multi-Head Self-Attention Weights\n"I love AI"',
#              fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig("assignment2_attention_heatmap.png", dpi=150, bbox_inches='tight')
# plt.show()

# print("\n✅  Both assignments complete.")
# print("    Graphs saved: assignment1_curves.png  |  assignment2_attention_heatmap.png")