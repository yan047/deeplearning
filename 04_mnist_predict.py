from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.models import load_model

# load model 
model = load_model('MNIST.h5')

# load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape test images
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# extend first image to 4 dimensions
import numpy as np
image = np.expand_dims( test_images[0] , axis=0)

# predict
classes = model.predict_classes( image )
print (classes)

probabilities = model.predict( image )
print (probabilities)