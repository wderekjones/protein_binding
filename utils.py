import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing.imputation import Imputer


def load_data(data_path, sample_size=None, mode=None):
    data = pd.read_csv(data_path)

    if mode is 0:
        data = data.loc[data.iloc[:, -1] == 0]

    elif mode is 1:
        data = data.loc[data.iloc[:, -1] == 1]

    if sample_size is None:
        sample_size = data.shape[0]

    data = data.sample(sample_size)
    data.replace(to_replace='na', value=np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.as_matrix()
    labels = data[:, (data.shape[1] - 1)]
    data = data[:, 0:-2]

    imputer = Imputer()
    data = imputer.fit_transform(data, labels)

    return data, labels


def combine_positive_negative_data(positive, negative):
    '''deprecating this soon'''
    data = np.concatenate([positive, negative], axis=0)
    return data


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def feature_selection():
    X_n, y_n = load_data("data/ml_pro_features_labels.csv", 194991, mode=0)
    X_p, y_p = load_data("data/ml_pro_features_labels.csv", 4760, mode=1)
    X = combine_positive_negative_data(X_n, X_p)
    y = combine_positive_negative_data(y_n, y_p)
    imputer = Imputer()
    X = imputer.fit_transform(X)

    feat_select = VarianceThreshold(threshold=1.0)
    feat = feat_select.fit_transform(X)
    return feat


def generate_report(report_title, clf, preds, y_test):
    '''
        Takes a classifier object and evaluates model performance, generates figures and a table of results
    :param clf: 
    :param X: 
    :param y: 
    :return: 
    '''

    output_file = open("results/" + report_title + ".txt", "w")
    output_file.write(report_title + "\n")
    output_file.write(classification_report(y_test, preds, target_names=["Class 0", "Class 1"], digits=4))
    output_file.write("\n" + clf.__str__())
    output_file.close()

    confusion = confusion_matrix(y_test, preds)
    plt.clf()
    plot_confusion_matrix(confusion, classes=[0, 1], title=report_title)
    plt.savefig("results/" + report_title + ".png")
