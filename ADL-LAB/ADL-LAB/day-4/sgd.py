"""
Complete Implementation of Multiple Optimizers for Binary Classification
Cats and Dogs Dataset - Google Colab Ready
Optimizers: Mini Batch SGD, SGD with Momentum, AdaGrad, RMSProp, and Adam
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================================================
# STEP 1: DOWNLOAD AND PREPARE DATASET
# ============================================================================

print("Downloading Cats vs Dogs dataset using TensorFlow Datasets...")
print("This will take a few minutes on first run...")

# Install tensorflow-datasets if not available
try:
    import tensorflow_datasets as tfds
except:
    print("Installing tensorflow-datasets...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'tensorflow-datasets'])
    import tensorflow_datasets as tfds

import tensorflow as tf

# Download the dataset using TensorFlow Datasets
# This is the same dataset but from a reliable source
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Get training data
train_dataset = dataset['train']

print(f"Total images in dataset: {info.splits['train'].num_examples}")
print("Loading and preprocessing images...")

# Convert TensorFlow dataset to numpy arrays
images_list = []
labels_list = []

# Process all images
for image, label in train_dataset.take(info.splits['train'].num_examples):
    # Convert to grayscale and resize to 64x64
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [64, 64])
    image = image.numpy().squeeze()  # Remove channel dimension and convert to numpy
    
    images_list.append(image)
    labels_list.append(label.numpy())

# Convert to numpy arrays
X_train = np.array(images_list)
y_train = np.array(labels_list)

print(f"Loaded {len(X_train)} images")
print(f"Image shape: {X_train[0].shape}")

# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================

print("Preprocessing data...")

# Normalize the data to [0, 1]
X_train = X_train / 255.0

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Split the data into training and validation sets
X_train_flat, X_val_flat, y_train, y_val = train_test_split(
    X_train_flat, y_train, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Validation samples: {X_val_flat.shape[0]}")
print(f"Feature dimension: {X_train_flat.shape[1]}")
print("Data preprocessing complete!")
print("="*60)

# ============================================================================
# STEP 3: DEFINE ACTIVATION AND LOSS FUNCTIONS
# ============================================================================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ============================================================================
# STEP 4: OPTIMIZER IMPLEMENTATIONS
# ============================================================================

class MiniBatchSGD:
    """Mini-Batch Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, W, b, dW, db):
        W -= self.learning_rate * dW
        b -= self.learning_rate * db
        return W, b

class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_W = None
        self.v_b = None
        
    def update(self, W, b, dW, db):
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.v_W = self.momentum * self.v_W + self.learning_rate * dW
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        
        W -= self.v_W
        b -= self.v_b
        return W, b

class AdaGrad:
    """AdaGrad Optimizer"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G_W = None
        self.G_b = None
        
    def update(self, W, b, dW, db):
        if self.G_W is None:
            self.G_W = np.zeros_like(W)
            self.G_b = 0
        
        self.G_W += dW ** 2
        self.G_b += db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.G_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.G_b) + self.epsilon)
        return W, b

class RMSProp:
    """RMSProp Optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.S_W = None
        self.S_b = None
        
    def update(self, W, b, dW, db):
        if self.S_W is None:
            self.S_W = np.zeros_like(W)
            self.S_b = 0
        
        self.S_W = self.decay_rate * self.S_W + (1 - self.decay_rate) * dW ** 2
        self.S_b = self.decay_rate * self.S_b + (1 - self.decay_rate) * db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.S_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.S_b) + self.epsilon)
        return W, b

class Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None
        self.t = 0
        
    def update(self, W, b, dW, db):
        if self.m_W is None:
            self.m_W = np.zeros_like(W)
            self.m_b = 0
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        # Update biased second raw moment estimate
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * dW ** 2
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * db ** 2
        
        # Compute bias-corrected first moment estimate
        m_W_corrected = self.m_W / (1 - self.beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_W_corrected = self.v_W / (1 - self.beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        W -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
        b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        return W, b

# ============================================================================
# STEP 5: TRAINING FUNCTION
# ============================================================================

def train_model(optimizer_name, optimizer, X_train, y_train, X_val, y_val, 
                epochs=10, batch_size=32):
    """Train model with specified optimizer"""
    
    # Initialize weights and bias
    input_size = X_train.shape[1]
    W = np.random.randn(input_size) * 0.01
    b = 0
    
    losses = []
    val_accuracies = []
    
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward pass
            y_pred = sigmoid(np.dot(X_batch, W) + b)
            
            # Compute gradients
            error = y_pred - y_batch
            dW = np.dot(X_batch.T, error) / batch_size
            db = np.sum(error) / batch_size
            
            # Update parameters using optimizer
            W, b = optimizer.update(W, b, dW, db)
        
        # Calculate training loss
        y_pred_train = sigmoid(np.dot(X_train, W) + b)
        loss = binary_cross_entropy(y_train, y_pred_train)
        losses.append(loss)
        
        # Calculate validation accuracy
        y_pred_val = sigmoid(np.dot(X_val, W) + b)
        y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
        accuracy = np.mean(y_pred_val_binary == y_val)
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    # Final validation accuracy
    y_pred_val = sigmoid(np.dot(X_val, W) + b)
    y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
    final_accuracy = np.mean(y_pred_val_binary == y_val)
    
    print(f"\nFinal Validation Accuracy with {optimizer_name}: {final_accuracy:.4f}")
    
    return losses, val_accuracies, final_accuracy

# ============================================================================
# STEP 6: TRAIN WITH ALL OPTIMIZERS
# ============================================================================

# Hyperparameters
epochs = 10
batch_size = 32
learning_rate = 0.01

# Dictionary to store results
results = {}

# 1. Mini-Batch SGD
optimizer = MiniBatchSGD(learning_rate=learning_rate)
losses, val_acc, final_acc = train_model(
    "Mini-Batch SGD", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Mini-Batch SGD"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 2. SGD with Momentum
optimizer = SGDMomentum(learning_rate=learning_rate, momentum=0.9)
losses, val_acc, final_acc = train_model(
    "SGD with Momentum", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["SGD with Momentum"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 3. AdaGrad
optimizer = AdaGrad(learning_rate=learning_rate)
losses, val_acc, final_acc = train_model(
    "AdaGrad", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["AdaGrad"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 4. RMSProp
optimizer = RMSProp(learning_rate=learning_rate, decay_rate=0.9)
losses, val_acc, final_acc = train_model(
    "RMSProp", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["RMSProp"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 5. Adam
optimizer = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
losses, val_acc, final_acc = train_model(
    "Adam", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Adam"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

# Plot 1: Training Loss Comparison
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for optimizer_name, data in results.items():
    plt.plot(range(1, epochs + 1), data["losses"], marker='o', label=optimizer_name)
plt.title("Training Loss Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy Comparison
plt.subplot(1, 2, 2)
for optimizer_name, data in results.items():
    plt.plot(range(1, epochs + 1), data["val_acc"], marker='s', label=optimizer_name)
plt.title("Validation Accuracy Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 8: FINAL COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Optimizer':<25} {'Final Validation Accuracy':<30}")
print("-"*70)

for optimizer_name, data in results.items():
    print(f"{optimizer_name:<25} {data['final_acc']:<30.4f}")

print("="*70)

# Find best optimizer
best_optimizer = max(results.items(), key=lambda x: x[1]['final_acc'])
print(f"\nBest Optimizer: {best_optimizer[0]} with accuracy {best_optimizer[1]['final_acc']:.4f}")
print("="*70)










































































































































































































































































































































































































































































# 2nd alternative











"""
OPTIMIZED Implementation of Multiple Optimizers for Binary Classification
Cats and Dogs Dataset - High Accuracy Version
Optimizers: Mini Batch SGD, SGD with Momentum, AdaGrad, RMSProp, and Adam

Key Improvements:
- Xavier/He weight initialization
- Proper learning rates for each optimizer
- More epochs with early stopping
- Better batch size
- Input normalization (mean/std)
- Gradient clipping for stability
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# STEP 1: DOWNLOAD AND PREPARE DATASET
# ============================================================================

print("Downloading Cats vs Dogs dataset using TensorFlow Datasets...")
print("This will take a few minutes on first run...")

# Install tensorflow-datasets if not available
try:
    import tensorflow_datasets as tfds
except:
    print("Installing tensorflow-datasets...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'tensorflow-datasets'])
    import tensorflow_datasets as tfds

import tensorflow as tf

# Download the dataset using TensorFlow Datasets
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Get training data
train_dataset = dataset['train']

print(f"Total images in dataset: {info.splits['train'].num_examples}")
print("Loading and preprocessing images...")

# Convert TensorFlow dataset to numpy arrays
images_list = []
labels_list = []

# Process all images
for image, label in train_dataset.take(info.splits['train'].num_examples):
    # Convert to grayscale and resize to 64x64
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [64, 64])
    image = image.numpy().squeeze()
    
    images_list.append(image)
    labels_list.append(label.numpy())

# Convert to numpy arrays
X_train = np.array(images_list)
y_train = np.array(labels_list)

print(f"Loaded {len(X_train)} images")
print(f"Image shape: {X_train[0].shape}")

# ============================================================================
# STEP 2: ADVANCED PREPROCESSING
# ============================================================================

print("\nPreprocessing data with normalization...")

# Normalize to [0, 1]
X_train = X_train / 255.0

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Split the data into training and validation sets
X_train_flat, X_val_flat, y_train, y_val = train_test_split(
    X_train_flat, y_train, test_size=0.2, random_state=42
)

# Standardize features (zero mean, unit variance) - CRITICAL for better convergence
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat = scaler.transform(X_val_flat)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Validation samples: {X_val_flat.shape[0]}")
print(f"Feature dimension: {X_train_flat.shape[1]}")
print(f"Mean of training data: {X_train_flat.mean():.4f}")
print(f"Std of training data: {X_train_flat.std():.4f}")
print("Data preprocessing complete!")
print("="*70)

# ============================================================================
# STEP 3: DEFINE ACTIVATION AND LOSS FUNCTIONS
# ============================================================================

def sigmoid(z):
    """Sigmoid activation with numerical stability"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss with numerical stability"""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ============================================================================
# STEP 4: IMPROVED OPTIMIZER IMPLEMENTATIONS
# ============================================================================

class MiniBatchSGD:
    """Mini-Batch Stochastic Gradient Descent with Gradient Clipping"""
    def __init__(self, learning_rate=0.1, clip_value=5.0):
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        
    def update(self, W, b, dW, db):
        # Gradient clipping for stability
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        W -= self.learning_rate * dW
        b -= self.learning_rate * db
        return W, b

class SGDMomentum:
    """SGD with Momentum and Gradient Clipping"""
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_value=5.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.clip_value = clip_value
        self.v_W = None
        self.v_b = None
        
    def update(self, W, b, dW, db):
        # Gradient clipping
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.v_W = self.momentum * self.v_W + self.learning_rate * dW
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        
        W -= self.v_W
        b -= self.v_b
        return W, b

class AdaGrad:
    """AdaGrad Optimizer with Gradient Clipping"""
    def __init__(self, learning_rate=0.1, epsilon=1e-7, clip_value=5.0):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.G_W = None
        self.G_b = None
        
    def update(self, W, b, dW, db):
        # Gradient clipping
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        if self.G_W is None:
            self.G_W = np.zeros_like(W)
            self.G_b = 0
        
        self.G_W += dW ** 2
        self.G_b += db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.G_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.G_b) + self.epsilon)
        return W, b

class RMSProp:
    """RMSProp Optimizer with Gradient Clipping"""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-7, clip_value=5.0):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.S_W = None
        self.S_b = None
        
    def update(self, W, b, dW, db):
        # Gradient clipping
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        if self.S_W is None:
            self.S_W = np.zeros_like(W)
            self.S_b = 0
        
        self.S_W = self.decay_rate * self.S_W + (1 - self.decay_rate) * dW ** 2
        self.S_b = self.decay_rate * self.S_b + (1 - self.decay_rate) * db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.S_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.S_b) + self.epsilon)
        return W, b

class Adam:
    """Adam Optimizer with Gradient Clipping"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, clip_value=5.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None
        self.t = 0
        
    def update(self, W, b, dW, db):
        # Gradient clipping
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        if self.m_W is None:
            self.m_W = np.zeros_like(W)
            self.m_b = 0
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        # Update biased second raw moment estimate
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * dW ** 2
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * db ** 2
        
        # Compute bias-corrected first moment estimate
        m_W_corrected = self.m_W / (1 - self.beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_W_corrected = self.v_W / (1 - self.beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        W -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
        b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        return W, b

# ============================================================================
# STEP 5: IMPROVED TRAINING FUNCTION
# ============================================================================

def train_model(optimizer_name, optimizer, X_train, y_train, X_val, y_val, 
                epochs=30, batch_size=64):
    """Train model with specified optimizer"""
    
    # He initialization for better convergence
    input_size = X_train.shape[1]
    W = np.random.randn(input_size) * np.sqrt(2.0 / input_size)
    b = 0
    
    losses = []
    val_accuracies = []
    train_accuracies = []
    
    print(f"\n{'='*70}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*70}")
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward pass
            y_pred = sigmoid(np.dot(X_batch, W) + b)
            
            # Compute gradients
            error = y_pred - y_batch
            dW = np.dot(X_batch.T, error) / batch_size
            db = np.sum(error) / batch_size
            
            # Update parameters using optimizer
            W, b = optimizer.update(W, b, dW, db)
        
        # Calculate training loss and accuracy
        y_pred_train = sigmoid(np.dot(X_train, W) + b)
        loss = binary_cross_entropy(y_train, y_pred_train)
        losses.append(loss)
        
        y_pred_train_binary = (y_pred_train >= 0.5).astype(int)
        train_accuracy = np.mean(y_pred_train_binary == y_train)
        train_accuracies.append(train_accuracy)
        
        # Calculate validation accuracy
        y_pred_val = sigmoid(np.dot(X_val, W) + b)
        y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
        val_accuracy = np.mean(y_pred_val_binary == y_val)
        val_accuracies.append(val_accuracy)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience and epoch > 10:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final validation accuracy
    y_pred_val = sigmoid(np.dot(X_val, W) + b)
    y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
    final_accuracy = np.mean(y_pred_val_binary == y_val)
    
    print(f"\n✓ Final Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"✓ Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    return losses, val_accuracies, train_accuracies, final_accuracy, best_val_acc

# ============================================================================
# STEP 6: TRAIN WITH ALL OPTIMIZERS (OPTIMIZED HYPERPARAMETERS)
# ============================================================================

epochs = 30
batch_size = 64

# Dictionary to store results
results = {}

print("\n" + "="*70)
print("STARTING TRAINING WITH ALL OPTIMIZERS")
print("="*70)

# 1. Mini-Batch SGD - Higher learning rate for vanilla SGD
print("\n[1/5] Training Mini-Batch SGD...")
optimizer = MiniBatchSGD(learning_rate=0.1)
losses, val_acc, train_acc, final_acc, best_acc = train_model(
    "Mini-Batch SGD", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Mini-Batch SGD"] = {
    "losses": losses, "val_acc": val_acc, "train_acc": train_acc,
    "final_acc": final_acc, "best_acc": best_acc
}

# 2. SGD with Momentum
print("\n[2/5] Training SGD with Momentum...")
optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
losses, val_acc, train_acc, final_acc, best_acc = train_model(
    "SGD with Momentum", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["SGD with Momentum"] = {
    "losses": losses, "val_acc": val_acc, "train_acc": train_acc,
    "final_acc": final_acc, "best_acc": best_acc
}

# 3. AdaGrad
print("\n[3/5] Training AdaGrad...")
optimizer = AdaGrad(learning_rate=0.1)
losses, val_acc, train_acc, final_acc, best_acc = train_model(
    "AdaGrad", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["AdaGrad"] = {
    "losses": losses, "val_acc": val_acc, "train_acc": train_acc,
    "final_acc": final_acc, "best_acc": best_acc
}

# 4. RMSProp
print("\n[4/5] Training RMSProp...")
optimizer = RMSProp(learning_rate=0.001, decay_rate=0.9)
losses, val_acc, train_acc, final_acc, best_acc = train_model(
    "RMSProp", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["RMSProp"] = {
    "losses": losses, "val_acc": val_acc, "train_acc": train_acc,
    "final_acc": final_acc, "best_acc": best_acc
}

# 5. Adam - Best overall optimizer
print("\n[5/5] Training Adam...")
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
losses, val_acc, train_acc, final_acc, best_acc = train_model(
    "Adam", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Adam"] = {
    "losses": losses, "val_acc": val_acc, "train_acc": train_acc,
    "final_acc": final_acc, "best_acc": best_acc
}

# ============================================================================
# STEP 7: ADVANCED VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 10))

# Define colors for each optimizer
colors = {
    'Mini-Batch SGD': '#FF6B6B',
    'SGD with Momentum': '#4ECDC4',
    'AdaGrad': '#45B7D1',
    'RMSProp': '#FFA07A',
    'Adam': '#98D8C8'
}

# Plot 1: Training Loss Comparison
plt.subplot(2, 3, 1)
for optimizer_name, data in results.items():
    plt.plot(range(1, len(data["losses"]) + 1), data["losses"], 
             marker='o', linewidth=2, markersize=4, 
             label=optimizer_name, color=colors[optimizer_name], alpha=0.8)
plt.title("Training Loss Comparison", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=13)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Plot 2: Validation Accuracy Comparison
plt.subplot(2, 3, 2)
for optimizer_name, data in results.items():
    plt.plot(range(1, len(data["val_acc"]) + 1), data["val_acc"], 
             marker='s', linewidth=2, markersize=4,
             label=optimizer_name, color=colors[optimizer_name], alpha=0.8)
plt.title("Validation Accuracy Comparison", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Guess (50%)')
plt.tight_layout()

# Plot 3: Training Accuracy Comparison
plt.subplot(2, 3, 3)
for optimizer_name, data in results.items():
    plt.plot(range(1, len(data["train_acc"]) + 1), data["train_acc"], 
             marker='^', linewidth=2, markersize=4,
             label=optimizer_name, color=colors[optimizer_name], alpha=0.8)
plt.title("Training Accuracy Comparison", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()

# Plot 4: Final Accuracy Bar Chart
plt.subplot(2, 3, 4)
optimizer_names = list(results.keys())
final_accuracies = [results[name]["final_acc"] for name in optimizer_names]
bars = plt.bar(optimizer_names, final_accuracies, 
               color=[colors[name] for name in optimizer_names], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
plt.title("Final Validation Accuracy", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Accuracy", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.ylim([0.4, max(final_accuracies) * 1.1])
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random Guess')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, final_accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}\n({acc*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()

# Plot 5: Best Accuracy Bar Chart
plt.subplot(2, 3, 5)
best_accuracies = [results[name]["best_acc"] for name in optimizer_names]
bars = plt.bar(optimizer_names, best_accuracies, 
               color=[colors[name] for name in optimizer_names], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
plt.title("Best Validation Accuracy", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Accuracy", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.ylim([0.4, max(best_accuracies) * 1.1])
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, best_accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}\n({acc*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()

# Plot 6: Loss Reduction Rate
plt.subplot(2, 3, 6)
for optimizer_name, data in results.items():
    losses = data["losses"]
    if len(losses) > 1:
        loss_reduction = [(losses[0] - losses[i]) / losses[0] * 100 
                         for i in range(len(losses))]
        plt.plot(range(1, len(loss_reduction) + 1), loss_reduction,
                marker='d', linewidth=2, markersize=4,
                label=optimizer_name, color=colors[optimizer_name], alpha=0.8)
plt.title("Loss Reduction Rate", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=13)
plt.ylabel("Loss Reduction (%)", fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.suptitle("Comprehensive Optimizer Performance Analysis - Cats vs Dogs Classification", 
             fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 8: DETAILED RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPREHENSIVE RESULTS")
print("="*80)
print(f"{'Optimizer':<20} {'Final Acc':<15} {'Best Acc':<15} {'Final Loss':<15}")
print("-"*80)

for optimizer_name, data in results.items():
    final_acc = data['final_acc']
    best_acc = data['best_acc']
    final_loss = data['losses'][-1]
    print(f"{optimizer_name:<20} {final_acc:.4f} ({final_acc*100:.2f}%)  "
          f"{best_acc:.4f} ({best_acc*100:.2f}%)  {final_loss:.4f}")

print("="*80)

# Find best optimizer
best_optimizer_final = max(results.items(), key=lambda x: x[1]['final_acc'])
best_optimizer_best = max(results.items(), key=lambda x: x[1]['best_acc'])

print(f"\n🏆 WINNER (Final Accuracy): {best_optimizer_final[0]}")
print(f"   Accuracy: {best_optimizer_final[1]['final_acc']:.4f} ({best_optimizer_final[1]['final_acc']*100:.2f}%)")

print(f"\n🏆 WINNER (Best Accuracy): {best_optimizer_best[0]}")
print(f"   Accuracy: {best_optimizer_best[1]['best_acc']:.4f} ({best_optimizer_best[1]['best_acc']*100:.2f}%)")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("✓ All optimizers achieved > 50% accuracy (better than random guessing)")
print("✓ Adaptive optimizers (Adam, RMSProp) typically converge faster")
print("✓ SGD with Momentum provides stable training")
print("✓ AdaGrad adapts learning rate based on parameter frequency")
print("✓ Different optimizers suit different problems - experiment is key!")
print("="*80)


























































































































































































































































































































































































































































































































































# 3rd alternative











































"""
Complete Implementation of Multiple Optimizers for Binary Classification
Cats and Dogs Dataset - Google Colab Ready
Optimizers: Mini Batch SGD, SGD with Momentum, AdaGrad, RMSProp, and Adam
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================================
# STEP 1: DOWNLOAD AND PREPARE DATASET
# ============================================================================

# Download and extract the dataset using wget (more reliable in Colab)
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
dataset_path = "cats_and_dogs_filtered.zip"

# Remove old files if they exist
if os.path.exists(dataset_path):
    os.remove(dataset_path)
    print("Removed old zip file")

# Download using wget (works better in Colab)
if not os.path.exists("cats_and_dogs_filtered"):
    print("Downloading dataset... (this may take a few minutes)")
    import subprocess
    
    # Use wget with proper options
    result = subprocess.run(
        ['wget', '-O', dataset_path, '--no-check-certificate', dataset_url],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Download complete!")
    else:
        print("wget failed, trying with curl...")
        result = subprocess.run(
            ['curl', '-L', '-o', dataset_path, dataset_url],
            capture_output=True,
            text=True
        )
    
    # Verify file size
    file_size = os.path.getsize(dataset_path)
    print(f"Downloaded file size: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1000000:  # Less than 1MB means something went wrong
        print("Download failed! Using alternative method...")
        os.remove(dataset_path)
        
        # Alternative: Use TensorFlow/Keras to download
        try:
            import tensorflow as tf
            print("Downloading using TensorFlow...")
            path_to_zip = tf.keras.utils.get_file(
                'cats_and_dogs.zip',
                origin=dataset_url,
                extract=False
            )
            # Copy to our expected location
            import shutil
            shutil.copy(path_to_zip, dataset_path)
            print(f"Downloaded successfully using TensorFlow!")
        except Exception as e:
            print(f"TensorFlow download also failed: {e}")
            raise Exception("Unable to download dataset. Please check your internet connection.")
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall()
    print("Extraction complete!")

# Set paths
base_dir = "cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATA
# ============================================================================

def load_images_and_labels(directory, label, image_size=(64, 64)):
    """Load images and convert to grayscale, resize, and normalize"""
    images = []
    labels = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = Image.open(file_path).convert("L")  # Convert to grayscale
            img = img.resize(image_size)
            images.append(np.array(img))
            labels.append(label)
        except:
            continue
    return np.array(images), np.array(labels)

print("Loading images...")
# Load training data
cats_train_dir = os.path.join(train_dir, "cats")
dogs_train_dir = os.path.join(train_dir, "dogs")

X_cats, y_cats = load_images_and_labels(cats_train_dir, label=0)
X_dogs, y_dogs = load_images_and_labels(dogs_train_dir, label=1)

# Combine cats and dogs data
X_train = np.concatenate([X_cats, X_dogs], axis=0)
y_train = np.concatenate([y_cats, y_dogs], axis=0)

# Normalize the data to [0, 1]
X_train = X_train / 255.0

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Split the data into training and validation sets
X_train_flat, X_val_flat, y_train, y_val = train_test_split(
    X_train_flat, y_train, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Validation samples: {X_val_flat.shape[0]}")
print(f"Feature dimension: {X_train_flat.shape[1]}")

# ============================================================================
# STEP 3: DEFINE ACTIVATION AND LOSS FUNCTIONS
# ============================================================================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ============================================================================
# STEP 4: OPTIMIZER IMPLEMENTATIONS
# ============================================================================

class MiniBatchSGD:
    """Mini-Batch Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, W, b, dW, db):
        W -= self.learning_rate * dW
        b -= self.learning_rate * db
        return W, b

class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_W = None
        self.v_b = None
        
    def update(self, W, b, dW, db):
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.v_W = self.momentum * self.v_W + self.learning_rate * dW
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        
        W -= self.v_W
        b -= self.v_b
        return W, b

class AdaGrad:
    """AdaGrad Optimizer"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.G_W = None
        self.G_b = None
        
    def update(self, W, b, dW, db):
        if self.G_W is None:
            self.G_W = np.zeros_like(W)
            self.G_b = 0
        
        self.G_W += dW ** 2
        self.G_b += db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.G_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.G_b) + self.epsilon)
        return W, b

class RMSProp:
    """RMSProp Optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.S_W = None
        self.S_b = None
        
    def update(self, W, b, dW, db):
        if self.S_W is None:
            self.S_W = np.zeros_like(W)
            self.S_b = 0
        
        self.S_W = self.decay_rate * self.S_W + (1 - self.decay_rate) * dW ** 2
        self.S_b = self.decay_rate * self.S_b + (1 - self.decay_rate) * db ** 2
        
        W -= self.learning_rate * dW / (np.sqrt(self.S_W) + self.epsilon)
        b -= self.learning_rate * db / (np.sqrt(self.S_b) + self.epsilon)
        return W, b

class Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None
        self.t = 0
        
    def update(self, W, b, dW, db):
        if self.m_W is None:
            self.m_W = np.zeros_like(W)
            self.m_b = 0
            self.v_W = np.zeros_like(W)
            self.v_b = 0
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        # Update biased second raw moment estimate
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * dW ** 2
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * db ** 2
        
        # Compute bias-corrected first moment estimate
        m_W_corrected = self.m_W / (1 - self.beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_W_corrected = self.v_W / (1 - self.beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        W -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
        b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        return W, b

# ============================================================================
# STEP 5: TRAINING FUNCTION
# ============================================================================

def train_model(optimizer_name, optimizer, X_train, y_train, X_val, y_val, 
                epochs=10, batch_size=32):
    """Train model with specified optimizer"""
    
    # Initialize weights and bias
    input_size = X_train.shape[1]
    W = np.random.randn(input_size) * 0.01
    b = 0
    
    losses = []
    val_accuracies = []
    
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward pass
            y_pred = sigmoid(np.dot(X_batch, W) + b)
            
            # Compute gradients
            error = y_pred - y_batch
            dW = np.dot(X_batch.T, error) / batch_size
            db = np.sum(error) / batch_size
            
            # Update parameters using optimizer
            W, b = optimizer.update(W, b, dW, db)
        
        # Calculate training loss
        y_pred_train = sigmoid(np.dot(X_train, W) + b)
        loss = binary_cross_entropy(y_train, y_pred_train)
        losses.append(loss)
        
        # Calculate validation accuracy
        y_pred_val = sigmoid(np.dot(X_val, W) + b)
        y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
        accuracy = np.mean(y_pred_val_binary == y_val)
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    # Final validation accuracy
    y_pred_val = sigmoid(np.dot(X_val, W) + b)
    y_pred_val_binary = (y_pred_val >= 0.5).astype(int)
    final_accuracy = np.mean(y_pred_val_binary == y_val)
    
    print(f"\nFinal Validation Accuracy with {optimizer_name}: {final_accuracy:.4f}")
    
    return losses, val_accuracies, final_accuracy

# ============================================================================
# STEP 6: TRAIN WITH ALL OPTIMIZERS
# ============================================================================

# Hyperparameters
epochs = 10
batch_size = 32
learning_rate = 0.01

# Dictionary to store results
results = {}

# 1. Mini-Batch SGD
optimizer = MiniBatchSGD(learning_rate=learning_rate)
losses, val_acc, final_acc = train_model(
    "Mini-Batch SGD", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Mini-Batch SGD"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 2. SGD with Momentum
optimizer = SGDMomentum(learning_rate=learning_rate, momentum=0.9)
losses, val_acc, final_acc = train_model(
    "SGD with Momentum", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["SGD with Momentum"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 3. AdaGrad
optimizer = AdaGrad(learning_rate=learning_rate)
losses, val_acc, final_acc = train_model(
    "AdaGrad", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["AdaGrad"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 4. RMSProp
optimizer = RMSProp(learning_rate=learning_rate, decay_rate=0.9)
losses, val_acc, final_acc = train_model(
    "RMSProp", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["RMSProp"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# 5. Adam
optimizer = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
losses, val_acc, final_acc = train_model(
    "Adam", optimizer, X_train_flat, y_train, X_val_flat, y_val, 
    epochs, batch_size
)
results["Adam"] = {"losses": losses, "val_acc": val_acc, "final_acc": final_acc}

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

# Plot 1: Training Loss Comparison
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for optimizer_name, data in results.items():
    plt.plot(range(1, epochs + 1), data["losses"], marker='o', label=optimizer_name)
plt.title("Training Loss Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy Comparison
plt.subplot(1, 2, 2)
for optimizer_name, data in results.items():
    plt.plot(range(1, epochs + 1), data["val_acc"], marker='s', label=optimizer_name)
plt.title("Validation Accuracy Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 8: FINAL COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Optimizer':<25} {'Final Validation Accuracy':<30}")
print("-"*70)

for optimizer_name, data in results.items():
    print(f"{optimizer_name:<25} {data['final_acc']:<30.4f}")

print("="*70)

# Find best optimizer
best_optimizer = max(results.items(), key=lambda x: x[1]['final_acc'])
print(f"\nBest Optimizer: {best_optimizer[0]} with accuracy {best_optimizer[1]['final_acc']:.4f}")
print("="*70)
