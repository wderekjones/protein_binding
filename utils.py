import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing.imputation import Imputer


def make_data(data_path, sample_size):
    data = pd.read_csv(data_path)
    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    return data, labels


def get_positive_data(data_path, sample_size):
    data = pd.read_csv(data_path)
    data = data.loc[data.iloc[:, -1] == 1]
    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    return data, labels


def get_negative_data(data_path, sample_size):
    data = pd.read_csv(data_path)
    data = data.loc[data.iloc[:, -1] == 0]
    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    return data, labels


def combine_positive_negative_data(positive, negative):
    data = np.concatenate([positive, negative], axis=0)
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
    plt.xticks(tick_marks, classes, rotation=45, fontsize=2)
    plt.yticks(tick_marks, classes, fontsize=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def feature_selection():
    X_n, y_n = get_negative_data("data/ml_pro_features_labels.csv", 194991)
    X_p, y_p = get_positive_data("data/ml_pro_features_labels.csv", 4760)
    X = combine_positive_negative_data(X_n, X_p)
    y = combine_positive_negative_data(y_n, y_p)
    imputer = Imputer()
    X = imputer.fit_transform(X)

    feat_select = VarianceThreshold(threshold=1.0)
    feat = feat_select.fit_transform(X)
    return feat
