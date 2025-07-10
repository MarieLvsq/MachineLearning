import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE

# ——— 1) Data prep ———
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=5000,
    stratify=y_train_full,
    random_state=42
)

# Convert y labels from (n, 1) → (n,) for tf.data compatibility
y_train = y_train.squeeze()
y_val = y_val.squeeze()

# ——— 2) Augmentation & resizing layers ———
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
resize_layer = tf.keras.layers.Resizing(224, 224)

# ——— 3) tf.data Datasets - All data processing is moved into parallel tf.data pipelines ———
def preprocess_train(x, y):
    x = resize_layer(x)
    x = data_augment(x)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x, y

def preprocess_val(x, y):
    x = resize_layer(x)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds
    .shuffle(10000)
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = (
    val_ds
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

# ——— 4) Build the model - no resizing ———
inp = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)(inp, training=False)

x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(10, activation="softmax")(x)

# No need to manually call .fit(x_train, y_train) — you pass in datasets instead
# Batching, shuffling, prefetching make training 2–10× faster, especially on CPU
model = tf.keras.Model(inputs=inp, outputs=out)

# ——— 5) Compile & train ———
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

# ——— 6) Plot results ———
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
