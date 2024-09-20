import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Moves encoding
move_encoding = {'R': 0, 'P': 1, 'S': 2}
move_decoding = {0: 'R', 1: 'P', 2: 'S'}

# Beat dictionary: R beats S, P beats R, S beats P
beat = { "R": "P", "P": "S", "S": "R" }

# Not lose strategy (50-50 chance to either win or tie)
not_lose = {
    "R": "PPR",  # Prefer Paper, but Rock is also a fallback to tie with Rock
    "P": "SSP",  # Prefer Scissors, fallback to tie with Paper
    "S": "RRS"   # Prefer Rock, fallback to tie with Scissors
}

# Updated data with 10 moves in each sequence
data = [
    (["R", "P", "S", "R", "P", "S", "R", "P", "S", "R"], "R"),
    (["P", "S", "R", "P", "S", "R", "P", "S", "R", "P"], "P"),
    (["S", "R", "S", "R", "P", "S", "R", "S", "P", "S"], "S"),
    (["R", "R", "S", "S", "P", "R", "P", "R", "S", "R"], "R"),
    (["P", "R", "P", "S", "S", "P", "R", "S", "P", "P"], "P"),
    (["S", "S", "R", "P", "R", "S", "P", "R", "S", "S"], "S"),
    (["R", "S", "R", "P", "P", "R", "S", "P", "R", "R"], "R"),
    (["P", "P", "R", "S", "S", "P", "R", "P", "R", "P"], "P"),
    (["R", "S", "R", "P", "R", "S", "P", "R", "S", "S"], "S"),
    (["S", "P", "P", "S", "R", "S", "R", "S", "P", "P"], "P"),
    # Add more sequences as needed...
]

# Convert data to training format
def prepare_data(data):
    X = []
    y = []
    for seq, next_move in data:
        if len(seq) == 10:
            X.append([move_encoding[move] for move in seq])
            y.append(move_encoding[next_move])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 10, 1))  # Reshape for LSTM (10 steps instead of 5)
    return X, y

X, y = prepare_data(data)

# Initialize empty move history for both players
my_history = []
opponent_history = []

# Model Building Function
def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(10, 1)))  # Input shape is (10, 1) for 10 moves
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 output classes: R, P, S
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=2)

# Save the model
model.save('rps.keras')

"""# Extract the data from history
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict.get('accuracy')  # May be 'acc' in some versions
val_accuracy = history_dict.get('val_accuracy')  # 'val_acc' in some versions

epochs = range(1, len(loss) + 1)

# Plot training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
if accuracy and val_accuracy:  # If accuracy data exists
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()"""
