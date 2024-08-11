# SAPFER: Emotion-Based Music Recommendation System

SAPFER (Smart Automated Playlists Based on Facial Emotion Recognition) recommends music based on your facial expressions using the FER 2013 dataset and Spotify API.

## Project Description

SAPFER uses a facial emotion recognition model trained on the FER 2013 dataset, capable of detecting 7 emotions. The system captures live video feed from a webcam, processes the feed to predict the current emotion, and then fetches a playlist of songs from Spotify through the Spotipy wrapper based on the detected emotion. The recommended songs are displayed on the screen.

## Features

- **Real-time Emotion Detection:** Live facial expression detection with immediate song recommendations.
- **Spotify Integration:** Playlists are dynamically fetched from Spotify using the API.
- **Modern UI:** Neumorphism-inspired user interface for an enhanced user experience.

## Running the App

To run the app locally, follow these steps:

1. Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Enter your Spotify Developer credentials in the `auth_manager` section of `spotipy.py`. This step is only necessary if you need to update recommendation playlists. Also, uncomment the import statement in `camera.py`.
3. Run the application:
    ```bash
    python app.py
    ```
4. Grant camera permission if prompted.

## Tech Stack

- **Keras**
- **TensorFlow**
- **Spotipy**
- **Tkinter** (For initial testing)
- **Flask**

## Dataset

The project utilizes the FER 2013 dataset, a well-known dataset for emotion recognition. This dataset enables the model to classify 7 distinct emotions. You can access the dataset [here](https://www.kaggle.com/msambare/fer2013).

**Note:** The dataset is highly imbalanced, with the "happy" class being overrepresented. This imbalance may contribute to the moderate accuracy achieved after training.

## Model Architecture

The model is a sequential neural network consisting of Conv2D, MaxPool2D, Dropout, Dense layers, and **Batch Normalization**:

1. **Conv2D Layers:** Filter sizes range from 32 to 128, with ReLU activation.
2. **Batch Normalization:** Applied after convolutional layers to normalize activations and stabilize training.
3. **Pooling Layers:** Pool size of (2,2).
4. **Dropout:** Set to 0.25 to avoid overfitting; higher values resulted in poorer performance.
5. **Dense Layer:** The final layer uses softmax activation for classifying the 7 emotions.

The model is optimized using the Adam optimizer, with `categorical_crossentropy` as the loss function and `accuracy` as the evaluation metric.

**Note:** Various architectures, including VGG16, were tested, but this model provided the best balance between accuracy and performance. Further tuning of hyperparameters could improve accuracy.

## Image Processing and Training

- Images were normalized, resized to (48,48), and converted to grayscale in batches of 64 using the Keras `ImageDataGenerator`.
- Training was conducted locally for 75 epochs, taking approximately 13 hours, resulting in an accuracy of around 66%.

## Current Status

The entire project is fully operational, with live detection providing smooth frame rates thanks to multithreading.

## Project Components

- **spotipy.py:** Handles connection to Spotify and retrieval of tracks using the Spotipy wrapper.
- **haarcascade:** Utilized for face detection.
- **camera.py:** Manages video streaming, frame capturing, emotion prediction, and song recommendation, passing data to `main.py`.
- **main.py:** The main Flask application file.
- **index.html:** The web page template, using basic HTML and CSS.
- **utils.py:** Utility module for threading, enabling real-time webcam detection.
- **train.py:** Script for image processing and training the model.
