import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(
        predictions,
        actual_values,
        label_names,
        normalize=False,
        title='Confusion Matrix',
        cmap=plt.cm.Blues
):
    """
    This function plots a confusion matrix.

    Args:
        predictions (array-like): Predicted labels returned by a classifier.
        actual_values (array-like): True labels for the test data.
        label_names (list): A list of strings for the target label names.
        normalize (bool, optional): If True, the confusion matrix will be normalized
                                    by dividing by the sum of each row. Defaults to False.
        title (str, optional): Title for the plot. Defaults to 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap, optional): Colormap for the plot.
                                                     Defaults to plt.cm.Blues.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(actual_values, predictions)

    if normalize:
        # Normalize the confusion matrix by dividing each row by its sum
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Print the confusion matrix
    print(cm)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set x and y axis ticks with label names
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45, ha="right")
    plt.yticks(tick_marks, label_names)

    # Add text annotations to the cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# --- Example Usage ---
if __name__ == '__main__':
    # Sample data for demonstration
    # Let's say we have 3 classes: 'Cat', 'Dog', 'Bird'
    true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2])
    predicted_labels = np.array([0, 1, 2, 0, 0, 2, 1, 1, 0, 2, 1, 0, 0, 1])
    class_names = ['Cat', 'Dog', 'Bird']

    print("--- Unnormalized Confusion Matrix ---")
    plot_confusion_matrix(
        predicted_labels,
        true_labels,
        class_names,
        title='Confusion Matrix (Unnormalized)'
    )

    print("\n--- Normalized Confusion Matrix ---")
    plot_confusion_matrix(
        predicted_labels,
        true_labels,
        class_names,
        normalize=True,
        title='Confusion Matrix (Normalized)'
    )

    # Another example with more classes
    true_labels_multi = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    predicted_labels_multi = np.array([0, 1, 2, 3, 0, 1, 1, 3, 0, 0, 2, 3, 0, 1, 2, 2, 0, 1, 2, 3])
    class_names_multi = ['Class A', 'Class B', 'Class C', 'Class D']

    print("\n--- Multi-class Normalized Confusion Matrix ---")
    plot_confusion_matrix(
        predicted_labels_multi,
        true_labels_multi,
        class_names_multi,
        normalize=True,
        title='Multi-class Confusion Matrix (Normalized)',
        cmap=plt.cm.Greens # Example with a different colormap
    )
