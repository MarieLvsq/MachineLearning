import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ——— 1) Data prep ———
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_full, x_test = x_train_full/255.0, x_test/255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=5000, 
    stratify=y_train_full, 
    random_state=42
)

# ——— 2) Augmentation & resizing ———
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
resize_layer = tf.keras.layers.Resizing(224, 224)

# ——— 3) Build the model ———
inp = tf.keras.Input(shape=(32, 32, 3))

#  a) Resize & augment
x = resize_layer(inp)                   # 32×32 → 224×224
x = data_augment(x)                     # only on train, Keras handles off for val/test

#  b) Preprocess & extract features
x = tf.keras.applications.resnet_v2.preprocess_input(x)
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling="avg"
)
base_model.trainable = False            # freeze for the first phase
x = base_model(x, training=False)

#  c) Your custom head
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inp, outputs=out)

# ——— 4) Compile & train ———
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
)

# ——— 5) Plot results ———
plt.plot(history.history["accuracy"],  label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
