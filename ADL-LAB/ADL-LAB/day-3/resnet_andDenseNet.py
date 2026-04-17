# =====================================================
# Implementing ResNet and DenseNet on LFW Dataset
# Single Python File (Notepad Ready)
# =====================================================

# -----------------------------
# STEP 1: INSTALL LIBRARIES
# -----------------------------
import os
os.system("pip install -q tensorflow numpy pandas matplotlib scikit-learn")

# -----------------------------
# STEP 2: IMPORT LIBRARIES
# -----------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

print("TensorFlow Version:", tf.__version__)

# -----------------------------
# STEP 3: LOAD LFW DATASET
# -----------------------------
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4)

X = lfw.images
y = lfw.target
class_names = lfw.target_names
num_classes = len(class_names)

print("Images shape:", X.shape)
print("Labels shape:", y.shape)
print("Number of classes:", num_classes)

# -----------------------------
# STEP 4: IMAGE PREPROCESSING
# -----------------------------
X_resized = np.array([
    tf.image.resize(img[..., np.newaxis], (224, 224)).numpy()
    for img in X
])

X_resized = np.repeat(X_resized, 3, axis=-1)
X_resized = X_resized / 255.0

print("Preprocessed shape:", X_resized.shape)

# -----------------------------
# STEP 5: TRAIN-VALIDATION SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_resized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)

# -----------------------------
# STEP 6: BUILD RESNET MODEL
# -----------------------------
def build_resnet(num_classes):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

resnet_model = build_resnet(num_classes)
resnet_model.summary()

# -----------------------------
# STEP 7: BUILD DENSENET MODEL
# -----------------------------
def build_densenet(num_classes):
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

densenet_model = build_densenet(num_classes)
densenet_model.summary()

# -----------------------------
# STEP 8: CALLBACKS
# -----------------------------
callbacks_resnet = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("resnet_lfw.h5", save_best_only=True)
]

callbacks_densenet = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("densenet_lfw.h5", save_best_only=True)
]

# -----------------------------
# STEP 9: TRAIN RESNET
# -----------------------------
history_resnet = resnet_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks_resnet
)

# -----------------------------
# STEP 10: TRAIN DENSENET
# -----------------------------
history_densenet = densenet_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks_densenet
)

# -----------------------------
# STEP 11: EVALUATION
# -----------------------------
resnet_loss, resnet_acc = resnet_model.evaluate(X_val, y_val)
densenet_loss, densenet_acc = densenet_model.evaluate(X_val, y_val)

print("ResNet Accuracy:", resnet_acc)
print("DenseNet Accuracy:", densenet_acc)

# -----------------------------
# STEP 12: PLOTTING RESULTS
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_resnet.history["accuracy"], label="Train")
plt.plot(history_resnet.history["val_accuracy"], label="Validation")
plt.title("ResNet Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_densenet.history["accuracy"], label="Train")
plt.plot(history_densenet.history["val_accuracy"], label="Validation")
plt.title("DenseNet Accuracy")
plt.legend()

plt.show()

