import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# ── (a) Download & Load MNIST Dataset ──────────────────────────────
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training samples : {x_train.shape[0]}")
print(f"Test samples     : {x_test.shape[0]}")
print(f"Image shape      : {x_train.shape[1:]}")

# Normalize pixel values 0-255 → 0.0-1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# ── Preview sample digits ───────────────────────────────────────────
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.suptitle("Sample digits from MNIST (0-9)")
plt.tight_layout()
plt.show()

# ── (b) Build the model ─────────────────────────────────────────────
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # 28x28 image → 784 inputs
    layers.Dense(128, activation='relu'),    # hidden layer
    layers.Dropout(0.2),                     # prevent overfitting
    layers.Dense(10, activation='softmax')   # 10 output classes (0-9)
])

model.summary()

# ── Compile ─────────────────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Train ───────────────────────────────────────────────────────────
print("\nTraining...\n")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# ── Evaluate on test set ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# ── Plot accuracy & loss curves ──────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ── Predict on first 10 test images ─────────────────────────────────
predictions = model.predict(x_test[:10])
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"P:{predictions[i].argmax()}\nA:{y_test[i]}")
    plt.axis('off')
plt.suptitle("Predicted (P) vs Actual (A)")
plt.tight_layout()
plt.show()

# ── Test with your own image ─────────────────────────────────────────
def predict_custom_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_array = image.img_to_array(img)

    # Invert: MNIST is white digit on black background
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape to (1, 28, 28)
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = prediction.argmax()
    confidence = prediction.max() * 100

    # Show result
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}  ({confidence:.1f}% confidence)")
    plt.axis('off')
    plt.show()

    print(f"Predicted digit: {predicted_digit}  |  Confidence: {confidence:.1f}%")

# ── Run on your image ────────────────────────────────────────────────
predict_custom_image("2.jpeg")