import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Loading data... ⌛")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training Data Shape: {train_images.shape}")
print(f"Number of Labels: {len(train_labels)}")

plt.figure()
plt.imshow(train_images[1], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.title(f"Label {train_labels[1]} ({class_names[train_labels[1]]})")
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model Built Seccessfully! ✅")

print("\nStarting Training... 🚀")

history = model.fit(train_images, train_labels, epochs=10)

print("Training Finished! ✅")

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

img_index = 75
img = test_images[img_index]
label = test_labels[img_index]

predictions = model.predict(np.array([img]))
predicted_label = np.argmax(predictions)

plt.figure()
plt.imshow(img, cmap=plt.cm.binary)
plt.title(f"Truth: {class_names[label]} | Prediction: {class_names[predicted_label]}")
plt.show()