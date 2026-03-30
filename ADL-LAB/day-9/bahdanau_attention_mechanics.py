# ============================================================
#  Bahdanau (Additive) Attention Mechanism — Full Assignment
#  Seq2Seq model + manual attention computation + visualizations
#  Compatible with Google Colab (TensorFlow 2.x)
# ============================================================

# ── 0. Install / import dependencies ────────────────────────
# In Colab, TensorFlow is pre-installed; matplotlib is also available.
# Uncomment if needed:
# !pip install tensorflow matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
#  PART 1 — Seq2Seq Model with Bahdanau Attention
# ============================================================

print("=" * 60)
print("PART 1 — Seq2Seq with Bahdanau Attention (Toy En→Fr)")
print("=" * 60)

# ── 1.1  Toy Dataset ─────────────────────────────────────────
input_texts  = ['i am happy', 'i love you', 'he is good']
target_texts = ['je suis heureux', 'je t aime', 'il est bon']

print("\nDataset:")
for src, tgt in zip(input_texts, target_texts):
    print(f"  {src!r:20s} → {tgt!r}")

# ── 1.2  Tokenization ────────────────────────────────────────
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)

input_seq  = tokenizer.texts_to_sequences(input_texts)
target_seq = tokenizer.texts_to_sequences(target_texts)

input_seq  = tf.keras.preprocessing.sequence.pad_sequences(input_seq,  padding='post')
target_seq = tf.keras.preprocessing.sequence.pad_sequences(target_seq, padding='post')

vocab_size    = len(tokenizer.word_index) + 1
embedding_dim = 64
units         = 128

print(f"\nVocab size    : {vocab_size}")
print(f"Input  shape  : {input_seq.shape}")
print(f"Target shape  : {target_seq.shape}")
print(f"\nWord → Index mapping:\n  {tokenizer.word_index}")

# ── 1.3  Encoder ─────────────────────────────────────────────
def build_encoder(vocab_size, embedding_dim, enc_units):
    """
    Simple sequential encoder:
    Embedding → GRU (return_sequences=True, return_state=True)
    Returns: (enc_output, enc_hidden_state)
    """
    inputs     = tf.keras.Input(shape=(None,))
    embedded   = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    output, state = tf.keras.layers.GRU(
        enc_units,
        return_sequences=True,
        return_state=True
    )(embedded)
    model = tf.keras.Model(inputs, [output, state], name="Encoder")
    return model

# ── 1.4  Bahdanau Attention Layer ────────────────────────────
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Additive (Bahdanau) Attention.

    score(s_{t-1}, h_i) = V · tanh(W1·h_i + W2·s_{t-1})
    α_i                 = softmax(score_i)
    context             = Σ α_i · h_i
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)   # applied to encoder outputs
        self.W2 = tf.keras.layers.Dense(units)   # applied to decoder hidden state
        self.V  = tf.keras.layers.Dense(1)       # scalar score

    def call(self, query, values):
        # query  : (batch, hidden_units)  → expand → (batch, 1, hidden_units)
        # values : (batch, seq_len, hidden_units)
        query_expanded  = tf.expand_dims(query, 1)

        # Additive score — broadcast over the sequence axis
        score           = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        # score : (batch, seq_len, 1)

        attention_weights = tf.nn.softmax(score, axis=1)
        # context : (batch, hidden_units)
        context_vector    = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.W1.units})
        return config

# ── 1.5  Decoder ─────────────────────────────────────────────
class Decoder(tf.keras.Model):
    """
    One-step decoder with Bahdanau Attention.
    At each step:
      1. Compute context vector via attention over encoder outputs
      2. Concatenate context with the current token embedding
      3. Run GRU for one step
      4. Project to vocab logits
    """

    def __init__(self, vocab_size, embedding_dim, units, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru       = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.fc        = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, hidden, enc_output):
        # x          : (batch, 1)           — current token
        # hidden     : (batch, units)        — previous decoder state
        # enc_output : (batch, src_len, units)

        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)                              # (batch, 1, emb_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1),
                        x], axis=-1)                       # (batch, 1, units+emb_dim)

        output, state = self.gru(x)                        # (batch, 1, units), (batch, units)
        output        = tf.squeeze(output, axis=1)         # (batch, units)
        logits        = self.fc(output)                    # (batch, vocab_size)

        return logits, state, attention_weights


# ── 1.6  Forward Pass ────────────────────────────────────────
print("\n" + "-" * 40)
print("Running forward pass …")

encoder = build_encoder(vocab_size, embedding_dim, units)
decoder = Decoder(vocab_size, embedding_dim, units)

sample_input          = tf.convert_to_tensor(input_seq)
enc_output, enc_hidden = encoder(sample_input)

batch_size = input_seq.shape[0]
# Seed decoder with the French word 'je' for all samples
dec_input  = tf.expand_dims([tokenizer.word_index['je']] * batch_size, 1)

logits, state, attn = decoder(dec_input, enc_hidden, enc_output)

print(f"Encoder output shape    : {enc_output.shape}  (batch, src_len, units)")
print(f"Encoder hidden shape    : {enc_hidden.shape}  (batch, units)")
print(f"Decoder logits shape    : {logits.shape}  (batch, vocab_size)")
print(f"Attention weights shape : {attn.shape}  (batch, src_len, 1)")


# ── 1.7  Visualise Attention Heatmap ─────────────────────────
def plot_attention_heatmap(attention, source_tokens, title, ax):
    """
    Renders a single attention heatmap on the given Axes object.
    attention : (src_len, 1) numpy array
    """
    attn_vals = attention[:, 0]   # shape: (src_len,)

    im = ax.imshow(
        attn_vals.reshape(1, -1),   # 1 × src_len for a horizontal bar
        cmap='viridis',
        aspect='auto',
        vmin=0.0, vmax=1.0
    )
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, fontsize=11, rotation=30, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(['α (decoder step 1)'], fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold', pad=8)
    return im


# Source vocab (index → word)
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

fig, axes = plt.subplots(1, batch_size,
                         figsize=(5 * batch_size, 3),
                         constrained_layout=True)

fig.suptitle("Bahdanau Attention Weights — Decoder Step 1 (seed = 'je')",
             fontsize=14, weight='bold')

for i in range(batch_size):
    src_tokens  = [index_to_word[idx]
                   for idx in input_seq[i] if idx != 0]   # strip padding
    attn_sample = attn[i].numpy()                         # (src_len, 1)

    im = plot_attention_heatmap(
        attn_sample, src_tokens,
        title=f'Sample {i+1}: "{input_texts[i]}"',
        ax=axes[i] if batch_size > 1 else axes
    )

cbar = fig.colorbar(im, ax=axes if batch_size > 1 else axes,
                    fraction=0.04, pad=0.04)
cbar.set_label('Attention weight', fontsize=10)

plt.savefig("attention_heatmap_seq2seq.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: attention_heatmap_seq2seq.png")


# ── 1.8  Bar-chart of attention for each sample ──────────────
fig2, axes2 = plt.subplots(1, batch_size,
                           figsize=(5 * batch_size, 4),
                           constrained_layout=True)

fig2.suptitle("Attention Weights — Bar Chart", fontsize=13, weight='bold')

for i in range(batch_size):
    ax = axes2[i] if batch_size > 1 else axes2
    src_tokens  = [index_to_word[idx]
                   for idx in input_seq[i] if idx != 0]
    attn_vals   = attn[i].numpy()[:len(src_tokens), 0]

    bars = ax.bar(src_tokens, attn_vals, color='steelblue', edgecolor='white')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Weight', fontsize=10)
    ax.set_title(f'"{input_texts[i]}"', fontsize=11, weight='bold')
    ax.tick_params(axis='x', rotation=20)

    for bar, val in zip(bars, attn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.2f}",
                ha='center', va='bottom', fontsize=9)

plt.savefig("attention_barchart_seq2seq.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: attention_barchart_seq2seq.png")


# ============================================================
#  PART 2 — Assignment (Slide 9): Manual Attention Computation
# ============================================================

print("\n" + "=" * 60)
print("PART 2 — Assignment: Manual Bahdanau Attention")
print("=" * 60)

# Given tensors
values = tf.constant([[[1.0, 2.0, 0.0],
                        [0.0, 1.0, 3.0],
                        [2.0, 0.0, 1.0]]])   # shape: (1, 3, 3)

query  = tf.constant([[1.0, 1.0, 1.0]])       # shape: (1, 3)

print(f"\nvalues shape : {values.shape}  → (batch=1, seq_len=3, depth=3)")
print(f"query  shape : {query.shape}   → (batch=1, depth=3)")
print(f"\nvalues:\n{values.numpy()}")
print(f"\nquery :\n{query.numpy()}")

# ── Step 1: instantiate and run attention ─────────────────────
attention_layer = BahdanauAttention(units=4)   # small units for clarity

context_vector, attention_weights = attention_layer(query, values)

print("\n" + "-" * 40)
print("STEP 1 — Score computation (W1·values + W2·query)")
print("-" * 40)

# Re-derive scores for display purposes
query_exp = tf.expand_dims(query, 1)                         # (1,1,3)
raw_score  = attention_layer.W1(values) + attention_layer.W2(query_exp)
tanh_score = tf.nn.tanh(raw_score)
score      = attention_layer.V(tanh_score)

print(f"  W1(values) + W2(query) shape : {raw_score.shape}")
print(f"  tanh(…)               shape  : {tanh_score.shape}")
print(f"  V(tanh(…)) — raw scores      :\n  {score.numpy().squeeze()}")

print("\n" + "-" * 40)
print("STEP 2 — Attention Weights (softmax over scores)")
print("-" * 40)
alpha = tf.nn.softmax(score, axis=1)
print(f"  Attention weights shape : {alpha.shape}")
print(f"  Attention weights :\n  {alpha.numpy().squeeze()}")
print(f"  Sum of weights = {tf.reduce_sum(alpha).numpy():.6f}  (must be 1.0)")

print("\n" + "-" * 40)
print("STEP 3 — Context Vector  (Σ α_i · h_i)")
print("-" * 40)
ctx = tf.reduce_sum(alpha * values, axis=1)
print(f"  Context vector shape : {ctx.shape}")
print(f"  Context vector : {ctx.numpy()}")

print("\n" + "-" * 40)
print("RESULTS SUMMARY")
print("-" * 40)
print(f"  Attention Scores  (raw) : {score.numpy().squeeze()}")
print(f"  Attention Weights (α)   : {attention_weights.numpy().squeeze()}")
print(f"  Context Vector          : {context_vector.numpy()}")


# ── Step 2: Visualise Part 2 Attention ───────────────────────
alpha_vals   = attention_weights.numpy().squeeze()   # (seq_len, 1) or (seq_len,)
if alpha_vals.ndim == 2:
    alpha_vals = alpha_vals[:, 0]

token_labels = [f"h{i+1}\n{values.numpy()[0, i]}" for i in range(3)]

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig3.suptitle("Assignment — Bahdanau Attention on Given values/query",
              fontsize=13, weight='bold')

# --- Heatmap ---
ax_heat = axes3[0]
im3 = ax_heat.imshow(
    alpha_vals.reshape(1, -1),
    cmap='YlOrRd', aspect='auto', vmin=0, vmax=1
)
ax_heat.set_xticks(range(3))
ax_heat.set_xticklabels([f"h{i+1}" for i in range(3)], fontsize=12)
ax_heat.set_yticks([0])
ax_heat.set_yticklabels(['α'], fontsize=12)
ax_heat.set_title("Heatmap of Attention Weights", fontsize=11)
for j, v in enumerate(alpha_vals):
    ax_heat.text(j, 0, f"{v:.3f}", ha='center', va='center',
                 fontsize=13, color='black', weight='bold')
fig3.colorbar(im3, ax=ax_heat, fraction=0.05)

# --- Bar chart ---
ax_bar = axes3[1]
colors = ['#e63946', '#457b9d', '#2a9d8f']
bars3  = ax_bar.bar([f"h{i+1}" for i in range(3)],
                    alpha_vals, color=colors, edgecolor='white', linewidth=1.2)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("Attention Weight (α)", fontsize=11)
ax_bar.set_title("Bar Chart of Attention Weights", fontsize=11)
ax_bar.axhline(1 / 3, color='grey', linestyle='--', linewidth=1,
               label='Uniform baseline (1/3)')
ax_bar.legend(fontsize=9)
for bar, val in zip(bars3, alpha_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, weight='bold')

plt.savefig("attention_assignment_part2.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: attention_assignment_part2.png")


# ── Step 3: Context vector composition ───────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 4), constrained_layout=True)

x         = np.arange(3)
width     = 0.22
alpha_np  = alpha_vals
vals_np   = values.numpy()[0]                  # (3, 3)
ctx_np    = context_vector.numpy()[0]          # (3,)

feat_colors = ['#264653', '#2a9d8f', '#e9c46a']
feat_labels = ['Feature 0', 'Feature 1', 'Feature 2']

for fi in range(3):
    contributions = alpha_np * vals_np[:, fi]   # α_i · h_i[fi]  for each h_i
    ax4.bar(x + fi * width, contributions,
            width=width, color=feat_colors[fi],
            alpha=0.85, label=feat_labels[fi],
            edgecolor='white')

ax4.set_xticks(x + width)
ax4.set_xticklabels([f"h{i+1}" for i in range(3)], fontsize=12)
ax4.set_ylabel("α_i · h_i[feature]", fontsize=11)
ax4.set_title(
    "Weighted Contributions to Context Vector  (Σ α_i · h_i = context)",
    fontsize=11, weight='bold'
)
ax4.legend(fontsize=10)

# Annotate final context values
for fi, val in enumerate(ctx_np):
    ax4.annotate(f"c[{fi}]={val:.3f}",
                 xy=(1.02, fi * 0.12 + 0.75),
                 xycoords='axes fraction',
                 fontsize=10, color=feat_colors[fi],
                 weight='bold')

plt.savefig("context_vector_composition.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: context_vector_composition.png")


# ============================================================
#  PART 3 — Explanation: How Bahdanau Attention Works
# ============================================================

print("\n" + "=" * 60)
print("PART 3 — Conceptual Explanation")
print("=" * 60)

explanation = """
Bahdanau (Additive) Attention — Key Concepts
─────────────────────────────────────────────
1. MOTIVATION
   Vanilla Seq2Seq compresses the entire source sentence into a single
   fixed-length context vector (the final encoder hidden state). This
   bottleneck hurts performance on long sentences.

2. CORE IDEA
   Instead of a single fixed vector, let the decoder "attend" to all
   encoder hidden states h₁, h₂, …, hₙ and learn which tokens to focus
   on at each decoding step.

3. SCORE FUNCTION  (additive / MLP-based)
   score(sₜ₋₁, hᵢ) = Vᵀ · tanh( W1·hᵢ + W2·sₜ₋₁ )
   • W1, W2, V are learned weight matrices.
   • sₜ₋₁ = previous decoder hidden state (the "query")
   • hᵢ   = encoder hidden state at position i (the "value/key")

4. ATTENTION WEIGHTS  (soft alignment)
   αᵢ = softmax( score(sₜ₋₁, hᵢ) )
   • αᵢ ≥ 0 and Σ αᵢ = 1  (probability distribution over source positions)
   • High α → decoder pays more attention to that source token.

5. CONTEXT VECTOR
   cₜ = Σᵢ αᵢ · hᵢ
   • A weighted sum of encoder states.
   • Passed to the decoder alongside the current token embedding.

6. DECODER STEP
   [cₜ ; embed(yₜ₋₁)]  →  GRU  →  Dense(vocab_size)  →  logits

7. ADVANTAGES OVER DOT-PRODUCT ATTENTION
   • Works well even when query and key have different dimensions.
   • More expressive: learns a non-linear compatibility function.
   • Original paper: Bahdanau et al., "Neural Machine Translation by
     Jointly Learning to Align and Translate" (ICLR 2015).
"""
print(explanation)

print("=" * 60)
print("All outputs saved. Assignment complete!")
print("=" * 60)


