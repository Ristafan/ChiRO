import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(
        predictions,
        actual_values,
        label_names,
        model_name,
        normalize=False,
        title='Confusion Matrix',
        cmap=plt.cm.Blues
):
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
    #plt.legend(title=model_name, loc='upper left')
    #plt.title(f'{title}')
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
