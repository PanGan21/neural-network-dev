import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data
from neural_network import LayerDense, ActivationReLu, ActivationLinear, LossMeanSquaredError, OptimizerAdam, AccuracyRegression
from model import Model


# Create dataset
X, y = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(1, 64))
model.add(ActivationReLu())
model.add(LayerDense(64, 64))
model.add(ActivationReLu())
model.add(LayerDense(64, 1))
model.add(ActivationLinear())

# Set loss and optimizer objects
model.set(loss=LossMeanSquaredError(), optimizer=OptimizerAdam(
    learning_rate=0.005, decay=1e-3), accuracy=AccuracyRegression())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100)
