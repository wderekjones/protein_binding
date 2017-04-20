import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def make_data(data_path):
    data = pd.read_csv(data_path,header=False)
    data = data.as_matrix()
    return data

def make_labels(labels_path):
    labels = pd.read_csv(labels_path, header=False)
    labels = labels.as_matrix()
    labels = labels[:,2]
    return labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=2)
    plt.yticks(tick_marks, classes,fontsize=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')