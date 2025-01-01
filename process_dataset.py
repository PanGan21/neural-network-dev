import numpy as np
import cv2
import os
from model import Model
from neural_network import LayerDense, ActivationReLu, ActivationSoftMax, LossCategoricalCrossentropy, OptimizerAdam, AccuracyCategorical


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(
                path, dataset, label, file
            ), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Flattening - Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5


# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLu())
model.add(LayerDense(128, 128))
model.add(ActivationReLu())
model.add(LayerDense(128, 10))
model.add(ActivationSoftMax())


# Set loss, optimizer and accuracy objects
model.set(loss=LossCategoricalCrossentropy(), optimizer=OptimizerAdam(
    decay=1e-3), accuracy=AccuracyCategorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

# Retrieve model parameters
parameters = model.get_parameters()

# New model
# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLu())
model.add(LayerDense(128, 128))
model.add(ActivationReLu())
model.add(LayerDense(128, 10))
model.add(ActivationSoftMax())

# Set loss and accuracy objects
# We do not set optimizer object this time - there's no need to do it
# as we won't train the model
model.set(
    loss=LossCategoricalCrossentropy(),
    accuracy=AccuracyCategorical())

# Finalize the model
model.finalize()

# Set model with parameters instead of training it
model.set_parameters(parameters)

# Evaluate the model
model.evaluate(X_test, y_test)

model.save_parameters('fashion_mnist.parms')
