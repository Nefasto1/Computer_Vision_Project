import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_statistics(statistics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot on the first subplot
    ax1.plot(range(len(statistics["Train Loss"])), statistics["Train Loss"], label="Train")
    ax1.plot(range(len(statistics["Test Loss"])), statistics["Test Loss"], label="Test")
    ax1.set_title("Losses")
    ax1.legend()
    
    # Plot on the second subplot
    ax2.plot(range(len(statistics["Train Accuracy"])), statistics["Train Accuracy"], label="Train")
    ax2.plot(range(len(statistics["Test Accuracy"])), statistics["Test Accuracy"], label="Test")
    ax2.set_title("Accuracies")
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def custom_confusion_matrix(ys, y_hats):
    classes = os.listdir("train/")
    
    cm = confusion_matrix(ys, y_hats)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Plot confusion matrix
    disp.plot(cmap=plt.cm.pink)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.show()