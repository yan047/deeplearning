from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# load images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#import matplotlib.pyplot as plt
#plt.subplot(111)
#plt.imshow(train_images[100], cmap=plt.get_cmap('gray'))
#plt.show()

# create deep network.
model = models.Sequential()

#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# 32 is a magic number, 32 kernels to apply, then 32 feature maps being generated.
model.add(layers.Conv2D(filters = 32, kernel_size =(3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# feed into classifier.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# reshape train images
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

# convert labels to categories.
train_labels = to_categorical(train_labels)

# start training.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


# prepare test data
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

# evaluate 
test_loss, test_acc = model.evaluate(test_images, test_labels)

test_acc
