import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# (1) Download & split into “full train” vs “test”
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# (2) Normalize pixel values from [0–255] → [0.0–1.0]
x_train_full = x_train_full.astype("float32") / 255.0
x_test       = x_test.astype("float32")       / 255.0

# (3) Split training into train vs validation (45k/5k)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=5000,
    stratify=y_train_full,
    random_state=42
)

# Only applied to training data
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# (1) Load a pretrained ResNet50V2 as backbone
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,       # drop the ImageNet classifier head
    weights="imagenet",      # load pretrained ImageNet weights
    input_shape=(32,32,3),
    pooling="avg"            # global average pooling at end → 1D vector
)
base_model.trainable = False  # freeze weights initially

# (2) Attach own head
inputs = tf.keras.Input(shape=(32,32,3))
x = data_augment(inputs)                     # apply augmentation
x = tf.keras.applications.resnet_v2.preprocess_input(x)
x = base_model(x, training=False)            # extract features
x = tf.keras.layers.Dropout(0.3)(x)          # regularization
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# --- A: Initial frozen training ---
history_frozen = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
)

# Plot to verify ~75–80% val accuracy
plt.plot(history_frozen.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch"); 
plt.ylabel("Accuracy"); 
plt.legend(); 
plt.show()


# --- B: Unfreeze + fine‑tune ---

# (1) Unfreeze last 50 layers of ResNet
for layer in base_model.layers[-50:]:
    layer.trainable = True

# (2) Recompile with smaller LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# (3) Add callbacks: early stop & LR reduction
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# (4) Fine tune
history_fine = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks
)

# Plot both train & val accuracy
plt.plot(history_fine.history["accuracy"], label="train acc")
plt.plot(history_fine.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()
