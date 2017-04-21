import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
need to select a subset of the data, make sure not to get the labels and examples confused out of order'''


def make_data(data_path,sample_size):
    data = pd.read_csv(data_path)
    data = data.sample(sample_size)
    data.replace(to_replace='na',value=np.nan,inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:,(data.shape[1]-1)]
    return data,labels


def get_positive_data(data_path,sample_size):
    data = pd.read_csv(data_path)
    data = data.loc[data.iloc[:, -1] == 1]
    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    return data, labels


def get_negative_data(data_path,sample_size):
    data = pd.read_csv(data_path)
    data = data.loc[data.iloc[:, -1] == 0]
    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    return data, labels

def combine_positive_negative_data(positive,negative):
    data = np.concatenate([positive,negative],axis=0)
    return data


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