import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    print("Loading raw dataset...")
    params = load_params()
    validation_size = params["preprocess"]["validation_size"]
    seed = params["preprocess"]["seed"]
    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")
    print("Normalizing pixel values...")
   # Main branch approach: convert first, then normalize in-place
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train /= 255.0
    x_test /= 255.0
    print("Splitting training and validation data...")
    x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=validation_size, random_state=seed,stratify=y_train)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)
    print("Preprocessing completed successfully.")
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Validation images:", x_val.shape)
    print("Validation labels:", y_val.shape)
    print("Test images:", x_test.shape)
    print("Test labels:", y_test.shape)

if __name__ == "__main__":
    main()