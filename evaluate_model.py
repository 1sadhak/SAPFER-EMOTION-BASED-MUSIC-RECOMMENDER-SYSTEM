from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load the emotion detection model
emotion_model = load_model('model.h5')

val_dir = 'data/test'
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

try:
    # Evaluate the pre-trained model on the validation data
    scores = emotion_model.evaluate_generator(val_generator, steps=len(val_generator))

    # Print the loss and accuracy
    print("Loss: %.4f" % scores[0])
    print("Accuracy: %.2f%%" % (scores[1] * 100))

except Exception as e:
    print("An error occurred during evaluation:", e)
