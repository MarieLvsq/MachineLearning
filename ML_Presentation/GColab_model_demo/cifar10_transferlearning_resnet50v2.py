# -*- coding: utf-8 -*-
"""CIFAR10_TransferLearning_ResNet50V2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10htp8kBLi2VW9qeAh8fa72JUOi395h4F
"""

# ✅ CIFAR-10 Transfer Learning with ResNet50V2 on Google Colab

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=5000, stratify=y_train_full, random_state=42
)
y_train = y_train.squeeze()
y_val = y_val.squeeze()
y_test = y_test.squeeze()

# Data augmentation and resizing
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
resize = tf.keras.layers.Resizing(224, 224)
AUTOTUNE = tf.data.AUTOTUNE

# tf.data pipelines
def preprocess_train(x, y):
    x = resize(x)
    x = data_augment(x)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x, y

def preprocess_val(x, y):
    x = resize(x)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x, y

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

# Build the model
inp = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")(inp)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs=inp, outputs=out)

# Phase 1: Train classifier head (frozen ResNet)
model.layers[1].trainable = False  # ResNet50V2

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_frozen = model.fit(train_ds, validation_data=val_ds, epochs=10);
model.save("frozen_model.h5");  # Save after phase 1

# Plot Phase 1
plt.plot(history_frozen.history["accuracy"], label="Train Acc (Frozen)")
plt.plot(history_frozen.history["val_accuracy"], label="Val Acc (Frozen)")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.show()

# Phase 2: Fine-tuning

# Load model and unfreeze top 50 layers
model = tf.keras.models.load_model("frozen_model.h5")
base_model = model.layers[1]  # ResNet50V2
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# Fine-tune
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# Plot Phase 2
plt.plot(history_fine.history["accuracy"], label="Train Acc (Fine-Tuned)")
plt.plot(history_fine.history["val_accuracy"], label="Val Acc (Fine-Tuned)")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.show()

# Evaluate on test set
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_val)
    .batch(128)
    .prefetch(AUTOTUNE)
)
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")