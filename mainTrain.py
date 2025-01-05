import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


image_directory='datasets/'

no_tumour_images=os.listdir(image_directory+ 'no/')
yes_tumour_images=os.listdir(image_directory+ 'yes/')

dataset=[]
label=[]
INPUT_SIZE=64


for i,image_name in enumerate(no_tumour_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in enumerate(yes_tumour_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2, random_state=0)

# Reshape=(n,image_width,image_height,n_channels)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)



# Model Building
# 64,64,3


# Define the input size
INPUT_SIZE = 64

# Initialize the Sequential model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))  # 32 filters, 3x3 kernel size, input shape defined
model.add(Activation('relu'))  # ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling with 2x2 pool size

# Second convolutional layer
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))  # 32 filters, 3x3 kernel size, He uniform initialization
model.add(Activation('relu'))  # ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling with 2x2 pool size

# Third convolutional layer
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))  # 64 filters, 3x3 kernel size, He uniform initialization
model.add(Activation('relu'))  # ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling with 2x2 pool size

# Flatten the feature maps into a 1D vector
model.add(Flatten())  # Flatten the input before inserting in the layers

# Fully connected layer
model.add(Dense(64))  # Dense layer with 64 neurons
model.add(Activation('relu'))  # ReLU activation function
model.add(Dropout(0.5))  # Dropout layer with 50% dropout rate to avoid overfitting

# Output layer
model.add(Dense(2))  # Dense layer with 1 neuron (output layer)
model.add(Activation('softmax'))  # Sigmoid activation function (for binary classification)

# model = Sequential([
#     Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),

#     Dense(64),
#     Activation('relu'),
#     Dropout(0.5),

#     Dense(2),
#     Activation('softmax')
# ])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric

# Train the model on the training data
model.fit(
    x_train,                  # The input data for training (features)
    y_train,                  # The target data for training (labels)
    batch_size=64,            # Number of samples per gradient update
    verbose=1,                # Verbosity mode: 1 means progress bar for each epoch
    epochs=50,               # Number of epochs to train the model
    validation_data=(x_test, y_test),  # Tuple of (input data, target data) for validation
    shuffle=False            # Whether to shuffle the training data before each epoch
)

# Save the trained model to a file
model.save('BrainTumor50EpochsCategorical.h5')  # Save the model architecture, weights, and training configuration to a file
