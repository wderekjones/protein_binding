import h5py
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import Imputer, Normalizer


def load_data(data_path, protein_name_list=None, sample_size=None, features_list=None, mode=None, conformation=None):
    input_fo = h5py.File(data_path, 'r')

    X = np.ndarray([], dtype=np.float32)
    y = np.ndarray([], dtype=np.float32)
    i = 0

    if protein_name_list is None:
        protein_name_list = list(input_fo.keys())
    print("loading", len(protein_name_list), "proteins.")
    for protein_name in tqdm(protein_name_list):
        x_, y_ = load_protein(data_path, protein_name=protein_name, sample_size=sample_size,
                              features_list=features_list, mode=mode, conformation=conformation)
        if i == 0:
            X = x_.astype(np.float32)
            y = y_.astype(np.float32)
        else:
            X = np.vstack((X, x_.astype(np.float32)))
            y = np.vstack((y, y_.astype(np.float32)))
        i += 1

    return X, y


def load_protein(data_path, protein_name=None, sample_size=None, features_list=None, mode=None, conformation=None):
    input_fo = h5py.File(data_path, 'r')

    # if features_list is none then use all of the features
    if features_list is None:
        features_list = list(input_fo[str(protein_name)].keys())
        features_list.remove("label")
        features_list.remove("receptor")
        features_list.remove("drugID")

        # in order to determine indices, select all of the labels and conformations, then seperately choose based on specifiedconditions, then find the intersection of the two sets.
    full_labels = np.asarray(input_fo[str(protein_name)]["label"]).flatten()
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
    data_array = np.zeros([sample_size, len(features_list)])
    i = 0

    for dataset in features_list:
        data = np.asarray(input_fo[str(protein_name)][str(dataset)], dtype=np.float32)[sample]
        data_array[:, i] = data[:, 0]
        i += 1

    label_array = np.asarray(input_fo[str(protein_name)]["label"])[sample]

    return data_array.astype(np.float32), label_array.astype(np.float32)

def generate_held_out_set(data_path,protein_name_list=None, features_list=None,mode=None,sample_size=None):
    input_fo = h5py.File(data_path)
    protein_list = list(input_fo.keys())
    holdout_protein = protein_list[0]
    proteins_to_load = protein_list
    for protein in protein_list:
        protein_list.remove(holdout_protein)
        print("holdout_protein : ",holdout_protein)
        X_train,y_train = load_data(data_path=data_path,protein_name_list=protein_list,features_list=features_list,mode=mode,sample_size=sample_size)
        X_test,y_test = load_protein(data_path=data_path,protein_name=holdout_protein,features_list=features_list,mode=mode,sample_size=sample_size)
        proteins_to_load.append(holdout_protein)
        holdout_protein = proteins_to_load[0]
        # should not do this everytime, makes things slow
        X_train = Normalizer().fit_transform(Imputer().fit_transform(X_train))
        X_test = Normalizer().fit_transform(Imputer().fit_transform(X_test))
        yield X_train,y_train.flatten(),X_test,y_test.flatten()