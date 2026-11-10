import os
import numpy as np
import pickle
from audio_utils import extract_mfcc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_model():
    X = []
    Y = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data")

    for user in os.listdir(data_path):
        user_folder = os.path.join(data_path, user)
        
        if not os.path.isdir(user_folder):
            continue

        for file in os.listdir(user_folder):
            file_path = os.path.join(user_folder, file)

            mfcc = extract_mfcc(file_path)

            X.append(mfcc)
            Y.append(user)

    if len(set(Y)) < 2:
        raise ValueError("At least two registered users are required.")

    if len(X) == 0:
        raise ValueError("No training audio found.")

    X = np.array(X)
    Y = np.array(Y)
    print("Training data shape:", X.shape)

# 🔥 Normalize data (VERY IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# 🔥 Train SVM
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_scaled, Y)

 # Save model + scaler
    with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("✅ SVM Model trained & saved!")

if __name__ == "__main__":
    train_model()