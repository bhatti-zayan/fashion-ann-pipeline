import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    print("Loading training parameters...")
    params = load_params()
    train_params = params["train"]
    dense_units = train_params["dense_units"]
    dropout_rate = train_params["dropout_rate"]
    learning_rate = train_params["learning_rate"]
    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]
    print("Loading processed training data...")
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    print("Building ANN model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile( optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    print("Starting model training...")
    history = model.fit(x_train,y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/history.csv", index=False)
    print("Training completed successfully.")
    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")

if __name__ == "__main__":
    main()