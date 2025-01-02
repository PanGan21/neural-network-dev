import matplotlib.pyplot as plt
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


fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

image_data = cv2.imread('./data/tshirt.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
predictions = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(predictions)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)


# Read an image
image_data = cv2.imread('./data/pants.png', cv2.IMREAD_GRAYSCALE)
# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))
# Invert image colors
image_data = 255 - image_data
# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
# Load the model
model = Model.load('fashion_mnist.model')
# Predict on the image
confidences = model.predict(image_data)
# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)
# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)
