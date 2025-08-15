# extract_features.py
import os, pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def extract_features(directory):
    base_model = VGG16()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    features = {}
    for name in os.listdir(directory):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[name.split('.')[0]] = feature.flatten()
    return features

if __name__ == "__main__":
    feats = extract_features("Flicker8k_Dataset")
    pickle.dump(feats, open("features.pkl", "wb"))
