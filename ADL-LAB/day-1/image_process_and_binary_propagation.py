# =========================================================
# GOOGLE COLAB FULL WORKING CODE - IMAGE PREPROCESS + BINARY CLASSIFICATION
# =========================================================

print("Starting setup...")

# -------------------- Install Required Libraries --------------------
! pip -q install pillow requests scikit-learn

# -------------------- Imports --------------------
import os
import zipfile
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file

print("Libraries imported successfully")

# =========================================================
# PART 1 — IMAGE PREPROCESSING DEMO
# =========================================================

print("\nLoading sample image...")

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/June_odd-eyed-cat.jpg/640px-June_odd-eyed-cat.jpg"
image_path = get_file("cat.jpg", origin=image_url)

img = Image.open(image_path).convert('RGB')

plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------- Resize --------------------
resized_img = img.resize((150,150))

plt.imshow(resized_img)
plt.title("Resized Image (150x150)")
plt.axis('off')
plt.show()

# -------------------- Normalize --------------------
normalized_img = np.array(resized_img) / 255.0

# -------------------- Augmentation --------------------
print("Generating augmentations...")

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

fig, axes = plt.subplots(3,3, figsize=(10,10))
for i, ax in enumerate(axes.flat):
    augmented_img = datagen.random_transform(normalized_img)
    ax.imshow(augmented_img)
    ax.set_title(f'Aug {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# -------------------- Crop --------------------
crop_box = (30,30,180,180)
cropped_img = resized_img.crop(crop_box)

plt.imshow(cropped_img)
plt.title("Cropped Image")
plt.axis('off')
plt.show()

# -------------------- Grayscale --------------------
grayscale_img = resized_img.convert("L")

plt.imshow(grayscale_img, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# =========================================================
# PART 2 — DOWNLOAD DATASET
# =========================================================

print("\nDownloading Cats vs Dogs dataset...")

dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "cats_and_dogs_filtered.zip"

if not os.path.exists(zip_path):
    r = requests.get(dataset_url)
    with open(zip_path, "wb") as f:
        f.write(r.content)

extract_path = "dataset"
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Dataset Ready!")

# Paths
base_dir = os.path.join(extract_path, "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# =========================================================
# DATA GENERATORS
# =========================================================

print("Preparing data generators...")

image_size = (150,150)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# =========================================================
# MODEL BUILDING
# =========================================================

print("Building model...")

model = Sequential([
    Flatten(input_shape=(150,150,3)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining started...")

epochs = 10   # Reduced for Colab speed

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# =========================================================
# EVALUATION
# =========================================================

print("\nEvaluating model...")
loss, accuracy = model.evaluate(validation_generator)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# =========================================================
# PLOTS
# =========================================================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\nALL TASKS COMPLETED SUCCESSFULLY 🎉")
