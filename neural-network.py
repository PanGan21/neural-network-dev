import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    """
    Defines a dense (fully connected) layer with weights and biases, 
    which takes n_inputs inputs and produces n_neurons outputs. 
    The forward method computes the output for a batch of inputs 
    by taking the dot product of inputs and weights and adding biases.
    """

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLu:
    """
    Implements the ReLU activation function, 
    which sets any negative values in the input to zero, 
    leaving positive values unchanged. 
    ReLU is commonly used in hidden layers for introducing non-linearity.
    """

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftMax:
    """
    This activation function is applied to the output layer of the network 
    to convert the raw output values into probabilities, 
    which are useful for classification tasks. It exponentiates each output, 
    normalizes by the total sum, and prevents overflow by subtracting the maximum value from inputs.
    """

    def forward(self, inputs):
        # prevent an overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    """
    The base Loss class includes a method to compute the mean loss over a batch.
    """

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    """
    Calculates the categorical cross-entropy loss, suitable for multi-class classification. 
    It calculates the negative log likelihood of the true class probabilities, 
    providing a measure of the network’s accuracy in classifying the input samples.
    """

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
activation1 = ActivationReLu()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = LossCategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(10000):
    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    # otherwise revert weights and biases
    if loss < lowest_loss:
        print('New set of weights found, iteration:', i,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
