import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# hardcoded path to csv
FER2013_CSV_PATH = "/home/rimil0bx/Documents/Projects/EMODetect/fer2013.csv"

def load_fer2013(csv_path=FER2013_CSV_PATH):
    """
    Load FER2013 dataset from CSV and preprocess images.
    
    Args:
        csv_path (str): Full path to fer2013.csv
        
    Returns:
        X_train, y_train, X_val, y_val : preprocessed numpy arrays
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FER2013 CSV not found at: {csv_path}\n"
                                f"Download it from Kaggle and place it here.")
    
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Emotion labels
    emotion_labels = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
    
    # Function to convert CSV pixel strings to numpy arrays
    def preprocess_pixels(pixels):
        pixels = np.array(pixels.split(), dtype='float32')
        pixels = pixels.reshape(48,48,1) 
        pixels /= 255.0  
        return pixels

    # Training set
    train_data = data[data['Usage']=='Training']
    X_train = np.array([preprocess_pixels(p) for p in tqdm(train_data['pixels'], desc="Preprocessing Training")])
    y_train = to_categorical(train_data['emotion'], num_classes=7)

    # Validation set
    val_data = data[data['Usage']=='PublicTest']
    X_val = np.array([preprocess_pixels(p) for p in tqdm(val_data['pixels'], desc="Preprocessing Validation")])
    y_val = to_categorical(val_data['emotion'], num_classes=7)

    return X_train, y_train, X_val, y_val

# example case
if __name__ == "__main__":
    try:
        X_train, y_train, X_val, y_val = load_fer2013()
        print("Training shape:", X_train.shape, y_train.shape)
        print("Validation shape:", X_val.shape, y_val.shape)
    except FileNotFoundError as e:
        print(e)
