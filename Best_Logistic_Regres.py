import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8],
              [5.1, 4.5], [6, 5], [5.6, 5],
              [3.3, 0.4], [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])

y = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])

np.random.seed(1)
input_neurons = 2
hidden_neurons_one = 3
hidden_neurons_two = 3
output_neurons = 1

W1 = 2 * np.random.random((input_neurons, hidden_neurons_one)) - 1
b1 = 2 * np.random.random((1, hidden_neurons_one)) - 1

W2 = 2 * np.random.random((hidden_neurons_one, hidden_neurons_two)) - 1
b2 = 2 * np.random.random((1, hidden_neurons_two)) - 1

W3 = 2 * np.random.random((hidden_neurons_two, output_neurons)) - 1
b3 = 2 * np.random.random((1, output_neurons))

learning_rate = 0.1

for i in range(10000):
    hidden_one_layer_input = np.dot(X, W1) + b1
    hidden_one_layer_output = sigmoid(hidden_one_layer_input)

    hidden_two_layer_input = np.dot(hidden_one_layer_output, W2) + b2
    hidden_two_layer_output = sigmoid(hidden_two_layer_input)

    hidden_three_layer_input = np.dot(hidden_two_layer_output, W3) + b3
    outputs = sigmoid(hidden_three_layer_input)

    error = y - outputs

    output_layer_adjustments = error * sigmoid_derivative(outputs)
    hidden_two_layer_error = np.dot(output_layer_adjustments, W3.T)
    hidden_two_layer_adjustments = hidden_two_layer_error * sigmoid_derivative(hidden_two_layer_output)
    hidden_one_layer_error = np.dot(hidden_two_layer_adjustments, W2.T)
    hidden_one_layer_adjustments = hidden_one_layer_error * sigmoid_derivative(hidden_one_layer_output)

    W3 += learning_rate * np.dot(hidden_two_layer_output.T, output_layer_adjustments)
    b3 += learning_rate * np.sum(output_layer_adjustments, axis=0)

    W2 += learning_rate * np.dot(hidden_one_layer_output.T, hidden_two_layer_adjustments)
    b2 += learning_rate * np.sum(hidden_two_layer_adjustments, axis=0)

    W1 += learning_rate * np.dot(X.T, hidden_one_layer_adjustments)
    b1 += learning_rate * np.sum(hidden_one_layer_adjustments, axis=0)

test_input = np.array([[2, 3]])
hidden_layer_test_one = sigmoid(np.dot(test_input, W1) + b1)
hidden_layer_test_two = sigmoid(np.dot(hidden_layer_test_one, W2) + b2)
output_test = sigmoid(np.dot(hidden_layer_test_two, W3) + b3)
print(output_test[0][0])