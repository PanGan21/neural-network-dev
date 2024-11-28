import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    """
    Represents a fully connected (dense) layer in a neural network.

    Attributes:
        weights (ndarray): Weight matrix for the layer.
        biases (ndarray): Bias vector for the layer.
        inputs (ndarray): Inputs to the layer during forward pass.
        output (ndarray): Output of the layer during forward pass.
        dweights (ndarray): Gradient of weights during backward pass.
        dbiases (ndarray): Gradient of biases during backward pass.
        dinputs (ndarray): Gradient of inputs during backward pass.
    """

    def __init__(self, n_inputs, n_neurons):
        """
        Initializes weights and biases for the dense layer.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of neurons in the layer.
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        """
        Performs the forward pass of the dense layer.

        Args:
            inputs (ndarray): Input data or outputs from the previous layer.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        """
        Performs the backward pass of the dense layer.

        Args:
            dvalues (ndarray): Gradients of the loss with respect to the layer's output.
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLu:
    """
    Implements the ReLU (Rectified Linear Unit) activation function.

    Attributes:
        inputs (ndarray): Inputs to the activation function during forward pass.
        output (ndarray): Outputs after applying ReLU.
        dinputs (ndarray): Gradients of inputs during backward pass.
    """

    def forward(self, inputs):
        """
        Applies ReLU activation to the input data.

        Args:
            inputs (ndarray): Input data or outputs from the previous layer.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        """
        Performs the backward pass for ReLU activation.

        Args:
            dvalues (ndarray): Gradients of the loss with respect to the activation's output.
        """
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    """
    Implements the SoftMax activation function.

    Attributes:
        inputs (ndarray): Inputs to the activation function during forward pass.
        output (ndarray): Probabilities after applying SoftMax.
        dinputs (ndarray): Gradients of inputs during backward pass.
    """

    def forward(self, inputs):
        """
        Applies SoftMax activation to the input data.

        Args:
            inputs (ndarray): Input data or outputs from the previous layer.
        """
        # Remember input values
        self.inputs = inputs
        # prevent an overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        """
        Performs the backward pass for SoftMax activation.

        Args:
            dvalues (ndarray): Gradients of the loss with respect to the activation's output.
        """
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    """
    Base class for all loss functions.

    Methods:
        calculate: Computes the mean loss over a batch of samples.
    """

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    """
    Implements categorical cross-entropy loss for classification tasks.
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

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalCrossentropy():
    """
    Combines SoftMax activation and categorical cross-entropy loss for efficiency.
    """

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = ActivationSoftMax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# SGD optimizer
class OptimizerSGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = ActivationReLu()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = LayerDense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
optimizer = OptimizerSGD()

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
