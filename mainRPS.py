import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

#encoding and decoding moves dictionaries
move_encoding = {'R': 0, 'P': 1, 'S': 2}
move_decoding = {0: 'R', 1: 'P', 2: 'S'}

#Winning moves
win = {
    "R": "P",
    "P": "S",
    "S": "R"
}
moves = ["R", "P", "S"]
#Loading the trained model
model = load_model('rps.keras')

#Initializing empty move history
my_history = []
opponent_history = []

#Predicting the opponent's next move
def predict_move(opponent_history):

    if len(opponent_history) < 5:
        return random.choice(["R", "P", "S"])

    latest_sequence = opponent_history[-5:]
    encoded_sequence = [move_encoding[move] for move in latest_sequence]
    encoded_sequence = np.reshape(encoded_sequence, (1, 5, 1))

    predicted_probs = model.predict(encoded_sequence)
    predicted_move_index = np.argmax(predicted_probs)
    predicted_move = move_decoding[predicted_move_index]

    return predicted_move

#Deciding which move to make next with the predicted move
def get_move(input_move, opponent_history):
    if input_move not in move_encoding:
        return random.choice(moves)

    opponent_history.append(input_move)
    predicted_move = predict_move(opponent_history)

    if predicted_move in win:
        return win[predicted_move]
    else:
        return random.choice(moves)

#Making the next move based on the opponent's last move.
def make_move(input):

    global my_history, opponent_history
    output = get_move(input, opponent_history)
    my_history.append(output)
    return output

output = make_move(input)

"""def test_bot_moves():
    test_inputs = ["R", "P", "S", "R", "P"]
    for move in test_inputs:
        print(f"Opponent Move: {move} -> Bot Move: {get_move(move, opponent_history)}")

# Run the test
test_bot_moves()"""