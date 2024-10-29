import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Generate sequences of digits
def generate_sequences(x, y, seq_length=5):
    sequences_x = []
    sequences_y = []
    for _ in range(50000):  # Generate more sequences for better training
        indices = np.random.randint(0, len(x), seq_length)
        sequences_x.append(x[indices])
        sequences_y.append(y[indices])
    return np.array(sequences_x), np.array(sequences_y)

seq_length = 5
x_train_seq, y_train_seq = generate_sequences(x_train, y_train, seq_length)
x_test_seq, y_test_seq = generate_sequences(x_test, y_test, seq_length)

# Build the model
model = Sequential([
    Input(shape=(seq_length, 28, 28, 1)),
    TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    TimeDistributed(Dense(32, activation='relu')),
    TimeDistributed(Dense(10, activation='softmax'))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_seq, y_train_seq, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_seq, y_test_seq)
print(f"Test accuracy: {test_accuracy}")

# Save the model
model.save('mnist_sequence_model.h5')
print("Model saved as 'mnist_sequence_model.h5'")
