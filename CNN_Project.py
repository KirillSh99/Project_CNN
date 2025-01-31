from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np

digits = fetch_olivetti_faces()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)


def convolutional(image, filter_weights, bias):
    image_height, image_width = image.shape
    filter_height, filter_width = filter_weights.shape

    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1

    output = np. zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            pixel_position = image[i:i+filter_height, j:j+filter_width]
            output[i, j] = np. sum(pixel_position*filter_weights) + bias.item()

    return output


def forward_pass(X_train, filter_weights, bias):
    outputs = []
    for image in X_train:
        output = convolutional(image, filter_weights, bias)
        outputs.append(output)
    return np.array(outputs)


def max_pooling(outputs, pool_size=2, stride=2):
    num_images, output_height, output_width = outputs.shape

    pooled_height = (output_height - pool_size) // stride + 1
    pooled_width = (output_width - pool_size) // stride + 1

    pooled_outputs = np.zeros((num_images, pooled_height, pooled_width))

    for img in range(num_images):
        for i in range(0, output_height - pool_size + 1, stride):
            for j in range(0, output_width - pool_size + 1, stride):

                block = outputs[img, i:i + pool_size, j:j + pool_size]

                pooled_outputs[img, i // stride, j // stride] = np.max(block)

    return pooled_outputs


def relu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < 0:
                x[i, j] = 0

    return x


filter_size = (3, 3)
filter_weights = np. random. rand(*filter_size) * 0.01
filter_bias = 2 * np.random.random(1) - 1

maps = forward_pass(X_train, filter_weights, filter_bias)
maps_relus = []
for i in maps:
    maps_relu = relu(i)
    maps_relus. append(maps_relu)

maps_relus = np. array(maps_relus)
maps_relus_pooling = max_pooling(maps_relus, pool_size=2, stride=2)
flattened = maps_relus_pooling.reshape(maps_relus_pooling.shape[0], -1)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(y_pred, y_true):
    return y_pred - y_true


num_classes = 40
y_train_one_hot = np.eye(num_classes)[y_train]

np.random.seed(1)
weights = 2 * np.random.random((flattened.shape[1], num_classes)) - 1
bias = 2 * np.random.random((1, num_classes)) - 1

learning_rate = 0.01

for i in range(1000):
    input_layer = flattened
    outputs = softmax(np.dot(input_layer, weights) + bias)

    error = softmax_derivative(outputs, y_train_one_hot)

    weights -= learning_rate * np.dot(input_layer.T, error)
    bias -= learning_rate * np.sum(error, axis=0, keepdims=True)

    filter_gradients = np.zeros_like(filter_weights)
    bias_gradient = 0

    for img_idx in range(X_train.shape[0]):
        for i in range(maps_relus_pooling.shape[1]):
            for j in range(maps_relus_pooling.shape[2]):
                for k in range(num_classes):
                    grad = error[img_idx, k]
                    filter_gradients += grad * X_train[img_idx][i:i + 3, j:j + 3]
                    bias_gradient += grad

    filter_weights -= learning_rate * filter_gradients
    filter_bias -= learning_rate * bias_gradient


maps_test = forward_pass(X_test, filter_weights, filter_bias)
maps_relus_test = []
for i in maps_test:
    maps_relu = relu(i)
    maps_relus_test. append(maps_relu)

maps_relus_test = np. array(maps_relus_test)
maps_relus_pooling_test = max_pooling(maps_relus_test, pool_size=2, stride=2)
flattened_test = maps_relus_pooling_test.reshape(maps_relus_pooling_test.shape[0], -1)
y_pred = softmax(np.dot(flattened_test, weights) + bias)
y_pred_classes = np.argmax(y_pred, axis=1)

correct_predictions = np.sum(y_test == y_pred_classes)
accuracy = correct_predictions / len(y_test)
print("Accuracy:", round(accuracy, 2))