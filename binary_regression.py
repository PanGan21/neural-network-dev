import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from neural_network import Loss, LayerDense, ActivationReLu, OptimizerAdam, ActivationSoftMax, LossCategoricalCrossentropy, AccuracyCategorical, LayerDropout
from model import Model

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(2, 512, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(ActivationReLu())
model.add(LayerDropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftMax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
    accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)
