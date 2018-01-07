import h5py
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import Imputer, Normalizer


def load_data(data_path, split=None, label=None, protein_name_list=None, sample_size=None, features_list=None, mode=None):

    input_fo = h5py.File(data_path, 'r')

    if split is not None:
        if split not in ["train", "test"]:
            print ("invalid split option")
            return None
        else:
            pass


    data_frame = pd.DataFrame()

    if protein_name_list is None:
        protein_name_list = list(input_fo[split].keys())
    print("loading", len(protein_name_list), "proteins.")
    if split is not None:
        for i, protein_name in enumerate(tqdm(protein_name_list)):
            protein_df = load_split_protein(data_path, label=label, protein_name=protein_name, split=split,
                                  sample_size=sample_size,
                                  features_list=features_list, mode=mode)
            data_frame = pd.concat([data_frame, protein_df])

    else:

        for i, protein_name in enumerate(tqdm(protein_name_list)):
            protein_df = load_protein(data_path, label=label, protein_name=protein_name, sample_size=sample_size,
                              features_list=features_list, mode=mode)
            data_frame = pd.concat([data_frame, protein_df])

    return data_frame

def load_protein(data_path, label=None, protein_name=None, sample_size=None, features_list=None, mode=None):
    input_fo = h5py.File(data_path, 'r')
    if label is None:
        label = "label"
    # if features_list is none then use all of the features
    if features_list is None:
        features_list = list(input_fo[str(protein_name)].keys())

    else:
        if "receptor" not in features_list:
            features_list.append("receptor")
        if "drugID" not in features_list:
            features_list.append("drugID")
        if "label" not in features_list:
            features_list.append("label")

    data_frame = pd.DataFrame()

    for idx, feature in enumerate(features_list):
        data_frame[feature] = np.ravel(input_fo[str(protein_name)+"/"+str(feature)])

    return data_frame



def load_split_protein(data_path, split=None, label=None, protein_name=None, sample_size=None, features_list=None, mode=None):
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

    if features_list is None:
        features_list = list(input_fo[split][str(protein_name)].keys())

    else:
        if "receptor" not in features_list:
            features_list.append("receptor")
        if "drugID" not in features_list:
            features_list.append("drugID")
        if "label" not in features_list:
            features_list.append("label")

    data_frame = pd.DataFrame()

    for idx, feature in enumerate(features_list):
        data_frame[feature] = np.ravel(input_fo[str(split)+"/"+str(protein_name)+"/"+str(feature)])

    return data_frame



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


def compute_intersection(small_list, big_list):
    intersect = set(big_list).intersection(set(small_list))
    return intersect


def compute_proportion(small_list, big_list):
    ct = len(compute_intersection(small_list,big_list))
    return ct/len(small_list)


def read_feature_list(feature_path):
    with open(feature_path, "r") as input_file:
        feature_list = []
        for line in input_file:
            line = line.strip('\n')
            feature_list.append(line)
    return feature_list


def output_feature_summary(output_path, full_feature_path, subset_feature_path, feature_list_root_path="/u/vul-d1/scratch/wdjo224/data/"):
    output_file = open(output_path, mode='w+')
    print("feature file: "+str(subset_feature_path),file=output_file)

    feature_subset = pd.read_csv(subset_feature_path, header=None)

    # load the full feature set
    full_features = pd.read_csv(full_feature_path, header=None)
    print("proportion of features in full feature set: "+str(compute_proportion(list(feature_subset[0]),list(full_features[0]))),file=output_file)

    # compute the number of features from the subset that belong to the binding group
    binding_features = pd.read_csv(feature_list_root_path+"binding_features_list.csv", header=None)
    print("proportion of features in binding feature set: "+str(compute_proportion(list(feature_subset[0]),list(binding_features[0]))),file=output_file)

    # compute the number of features from the subset that belong to the drug group
    dragon_features = pd.read_csv(feature_list_root_path+"drug_features_list.csv", header=None)
    print("proportion of features in dragon feature set: "+str(compute_proportion(list(feature_subset[0]), list(dragon_features[0]))),file=output_file)

    # compute the number of features from the subset that belong to the protein group
    drugminer_features = pd.read_csv(feature_list_root_path+"protein_features_list.csv", header=None)
    print("proportion of features in drug_miner feature set: "+str(compute_proportion(list(feature_subset[0]), list(drugminer_features[0]))),file=output_file)

    # compute the number of features from the subset that belong to the binding pocket group
    prank_features = pd.read_csv(feature_list_root_path+"pocket_features_list.csv", header=None)
    print("proportion of features in pocket feature set: "+str(compute_proportion(list(feature_subset[0]), list(prank_features[0]))),file=output_file)


def get_feature_histogram(feature_list, protein_feature_path=None, pocket_feature_path=None, drug_feature_path=None, binding_feature_path=None):
    feature_histogram = []
    protein_features = read_feature_list(protein_feature_path)
    pocket_features = read_feature_list(pocket_feature_path)
    drug_features = read_feature_list(drug_feature_path)
    binding_features = read_feature_list(binding_feature_path)

    for feature in feature_list:
        if feature in protein_features:
            feature_histogram.append("Protein")
        elif feature in pocket_features:
            feature_histogram.append("Pocket")
        elif feature in drug_features:
            feature_histogram.append("Drug")
        elif feature in binding_features:
            feature_histogram.append("Binding")

    return pd.DataFrame({"feature_class": feature_histogram})
