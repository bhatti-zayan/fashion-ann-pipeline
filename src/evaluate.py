import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    print("Loading trained model...")
    model = tf.keras.models.load_model("models/model.h5")
    print("Loading processed test data...")
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test,y_test,verbose=0)
    predictions = model.predict( x_test,verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(y_test, predicted_classes)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(9, 9))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()
    print("Evaluation completed successfully.")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to reports/confusion_matrix.png")

if __name__ == "__main__":
    main()