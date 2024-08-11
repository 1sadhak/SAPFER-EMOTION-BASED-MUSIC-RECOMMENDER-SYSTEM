import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# Define the emotions and their corresponding labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotions)

# Define the path to the test dataset
test_data_path = 'data/test'

# Initialize lists to store images and labels
test_images = []
test_labels = []

# Loop through each emotion folder
for i, emotion in enumerate(emotions):
    emotion_folder = os.path.join(test_data_path, emotion)
    for image_name in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_name)
        image = cv2.imread(image_path, 0)  # Read the image in grayscale
        image = cv2.resize(image, (48, 48))  # Resize the image to match the input shape of the model
        test_images.append(image)
        test_labels.append(i)  # Use the emotion label as the class index

# Convert the image and label lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Preprocess the images
test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
test_images = test_images.astype('float32')
test_images /= 255.0

# Convert the labels to one-hot encoded vectors
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Create the model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load the model weights
emotion_model.load_weights('model.h5')

# Compile the model
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test dataset
loss, accuracy = emotion_model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)