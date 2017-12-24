import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, normalize
from imblearn.over_sampling import SMOTE, RandomOverSampler


def load_data(data_path, split=None, label=None, protein_name_list=None, sample_size=None, features_list=None, mode=None):

    input_fo = h5py.File(data_path, 'r')

    if split is not None:
        if split not in ["train", "test"]:
            print ("caught exception")
            return None
        else:
            pass
    if split is None:
        print("must supply a split")
        return None

    X = np.ndarray([], dtype=np.float16)
    # use smaller precision for labels
    y = np.ndarray([], dtype=np.int8)
    i = 0

    if protein_name_list is None:
        protein_name_list = list(input_fo[split].keys())
    print("loading", len(protein_name_list), "proteins.")
    for i, protein_name in enumerate(tqdm(protein_name_list)):
        x_, y_ = load_protein(data_path, split=split, label=label, protein_name=protein_name, sample_size=sample_size,
                              features_list=features_list, mode=mode)
        if i == 0:
            X = x_
            y = y_
        else:
            X = np.vstack((X, x_))
            y = np.vstack((y, y_))

    return X,y


def load_protein(data_path, split=None, label=None, protein_name=None, sample_size=None, features_list=None, mode=None):
    if split is not None:
        if split not in ["train", "test"]:
            print ("caught exception")
            return None
        else:
            pass
    if split is None:
        print("must supply a split")
        return None
    input_fo = h5py.File(data_path, 'r')

    # check if user specified a feature to use as a label
    if label is None:
        # if not just use binding label
        label = "label"
    # if features_list is none then use all of the features
    if features_list is None:
        features_list = list(input_fo[split][str(protein_name)].keys())
        if label in features_list:
            features_list.remove(label)
        if "receptor" in features_list:
            features_list.remove("receptor")
        if "drugID" in features_list:
            features_list.remove("drugID")
        if "Filename" in features_list:
            features_list.remove("Filename")

        # in order to determine indices, select all of the labels, then seperately choose based on specifiedconditions, then find the intersection of the two sets.
    full_labels = np.asarray(input_fo[split][str(protein_name)][label]).flatten()
    full_idxs = np.arange(0, full_labels.shape[0], 1)

    mode_idxs = []
    if mode is not None:
        mode_idxs = full_idxs[full_labels[:, ] == mode]
    else:
        mode_idxs = full_idxs

    full_idxs = np.intersect1d(mode_idxs, full_idxs)

    # if sample size is none then select all of the indices
    if sample_size is None or sample_size > len(full_idxs):
        sample_size = len(full_idxs)

    sample = np.sort(np.random.choice(full_idxs, sample_size, replace=False))

    # get the data and store in numpy array
    data_array = np.zeros([sample_size, len(features_list)],dtype=np.float16)


    for idx, feature in enumerate(features_list):
        data_array[:,idx] = np.ravel(input_fo[str(split)+"/"+str(protein_name)+"/"+str(feature)])[sample]

    label_array = np.array(input_fo[str(split)+"/"+str(protein_name)+"/"+str(label)],dtype=np.int8)[sample]

    return data_array.astype(np.float16), label_array