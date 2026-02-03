from keras.models import load_model

# Load a pre-trained ECG classification model from the specified path
def load_ecg_model(path: str):
    return load_model(path)