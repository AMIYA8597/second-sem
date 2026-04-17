# =========================================
# LeNet-5 on MNIST Dataset (Jupyter Ready)
# =========================================

# ---- Step 0: Suppress TensorFlow CPU logs (important for Anaconda) ----
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---- Step 1: Import Libraries ----
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow Version:", tf.__version__)

# ---- Step 2: Load MNIST Dataset ----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# ---- Step 3: Visualize Sample Images ----
plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ---- Step 4: Data Preprocessing ----
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ---- Step 5: Padding (28x28 → 32x32) ----
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), mode='constant')

print("Padded training shape:", x_train.shape)

# ---- Step 6: LeNet-5 Model ----
model = models.Sequential([
    layers.Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1)),
    layers.AveragePooling2D(pool_size=(2,2), strides=2),

    layers.Conv2D(16, (5,5), activation='relu'),
    layers.AveragePooling2D(pool_size=(2,2), strides=2),

    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# ---- Step 7: Model Summary ----
model.summary()

# ---- Step 8: Compile ----
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---- Step 9: Train ----
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ---- Step 10: Evaluate ----
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ---- Step 11: Accuracy & Loss Graphs ----
plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.show()
