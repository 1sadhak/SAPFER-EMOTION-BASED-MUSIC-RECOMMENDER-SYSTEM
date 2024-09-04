from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class LogCallback(Callback):
    def __init__(self, filename='training_log.txt'):
        self.filename = filename
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch + 1} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}\n")
        self.train_loss.append(logs['loss'])
        self.train_acc.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_acc.append(logs['val_accuracy'])

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Print the model summary
emotion_model.summary()

log_callback = LogCallback()

from tensorflow.keras.callbacks import EarlyStopping

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=100,
    validation_data=val_generator,
    validation_steps=7178 // 64,
    callbacks=[log_callback, EarlyStopping(patience=5, restore_best_weights=True)]
)


emotion_model.save_weights('model.h5')

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(log_callback.train_acc) + 1), log_callback.train_acc, label='Training Accuracy')
plt.plot(range(1, len(log_callback.val_acc) + 1), log_callback.val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(log_callback.train_loss) + 1), log_callback.train_loss, label='Training Loss')
plt.plot(range(1, len(log_callback.val_loss) + 1), log_callback.val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
