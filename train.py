import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pickle

# Load the CIFAR-10 dataset from the local directory
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load training data
x_train_all = []
y_train_all = []
for i in range(1, 6):
    data_dict = unpickle(f'cifar-10-batches-py/data_batch_{i}')
    x_train_batch = data_dict[b'data']
    y_train_batch = data_dict[b'labels']
    x_train_all.append(x_train_batch)
    y_train_all.append(y_train_batch)

x_train = np.concatenate(x_train_all, axis=0)
y_train = np.concatenate(y_train_all, axis=0)

# Load test data
test_dict = unpickle('cifar-10-batches-py/test_batch')
x_test = test_dict[b'data']
y_test = test_dict[b'labels']

# Reshape input data
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model architecture with Leaky ReLU
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu))
model.add(layers.Conv2D(128, (3, 3), activation=tf.nn.leaky_relu))
model.add(layers.Conv2D(128, (3, 3), activation=tf.nn.leaky_relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation=tf.nn.leaky_relu))
model.add(layers.Dense(64, activation=tf.nn.leaky_relu))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

model.save('model/classification_model.h5')
print("Model saved successfully.")
