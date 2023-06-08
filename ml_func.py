import tensorflow as tf
import cv2
import numpy as np
from helpers import file_manager


# Load your trained model
model = tf.keras.models.load_model('ml/saved_model/lion_vs_tiger_ml_beta')


def makePrediction(filename):
    image_path = file_manager.UPLOAD_FOLDER + filename
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0

    prediction = model.predict(np.expand_dims(image, axis=0))

    class_labels = ["Lion", "Tiger"]
    predicted_label = class_labels[int(prediction[0][0] >= 0.5)]
    confidence = round(100 * (np.max(prediction[0])), 2)

    return {
        "confidence": confidence,
        "predicted_label": predicted_label
    }
