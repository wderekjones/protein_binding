import itertools
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing.imputation import Imputer
from functools import reduce


def load_data_csv(data_path, sample_size=None, mode=None):
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
    data = data[:, 0:-1]

    imputer = Imputer()
    data = imputer.fit_transform(data, labels)

    return data, labels


def load_data_h5(data_path, sample_size=None, features_list=None, mode=None, conformation=None):
    input_fo = h5py.File(data_path, 'r')

    dataset_size = input_fo["label"].shape[0]

    # if sample size is none then select all of the data
    if sample_size is None:
        sample_size = dataset_size

    # if features_list is none then use all of the features
    if features_list is None:
        features_list = list(input_fo.keys())
        features_list.remove("label")

    full_labels = np.asarray(input_fo["label"])

    j = 0
    idxs = []
    for element in full_labels:
        if mode is not None:
            if element[0] == mode:
                idxs.append(j)
        else:
            idxs.append(j)
        j += 1

    if sample_size > len(idxs):
        sample_size = len(idxs)
    sample = np.random.choice(idxs, sample_size, replace=False)

    # get the data and store in numpy array
    data_array = np.zeros([sample_size, len(features_list)])
    i = 0
    for dataset in features_list:
        data = np.asarray(input_fo[dataset])
        data = np.reshape(data, [data.shape[0]])
        data = data[sample]
        data_array[:, i] = data
        i += 1

    label_array = np.asarray(input_fo["label"])[sample]

    imputer = Imputer()
    data_array = imputer.fit_transform(data_array, label_array)

    return data_array, label_array


def combine_positive_negative_data(positive, negative):
    # TODO: take a list of positives and negatives as args
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
    X_n, y_n = load_data_csv("data/ml_pro_features_labels.csv", 194991, mode=0)
    X_p, y_p = load_data_csv("data/ml_pro_features_labels.csv", 4760, mode=1)
    X = combine_positive_negative_data(X_n, X_p)
    y = combine_positive_negative_data(y_n, y_p)
    imputer = Imputer()
    X = imputer.fit_transform(X)

    feat_select = VarianceThreshold(threshold=1.0)
    feat = feat_select.fit_transform(X)
    return feat


def generate_report(report_title, clf, preds, y_test):
    '''
    
    :param report_title: a string that will be used as the report filename 
    :param clf: a scikit learn classifier object
    :param preds: the predictions generated from the classifier 
    :param y_test: the counterpart ground truth values
    :return: nothing, outputs a file and generates corresponding confusion matrix
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
