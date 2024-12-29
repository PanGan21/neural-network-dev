import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from neural_network import Loss, LayerDense, ActivationReLu, OptimizerAdam, ActivationSigmoid, LossBinaryCrossentropy, AccuracyCategorical
from model import Model

nnfs.init()

# Create train and test dataset
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Instantiate the model
model = Model()
# Add layers
model.add(LayerDense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(LayerDense(64, 1))
model.add(ActivationReLu())
model.add(ActivationSigmoid())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossBinaryCrossentropy(), optimizer=OptimizerAdam(decay=5e-7), accuracy=AccuracyCategorical(binary=True)
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)
