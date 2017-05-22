import argparse
import pandas as pd
import numpy as np
import h5py
import time
from functools import reduce

# TODO: add metadata output

parser = argparse.ArgumentParser(description="Process files containing protein binding affinity features")

parser.add_argument('--p', type=str, nargs='+', help="list paths of files containing protein features")

parser.add_argument('--pm', type=str, help="list paths of files containing protein-molecular compound features")

parser.add_argument('--m', type=str, help="file containing molecular descriptors")

parser.add_argument('--feats', type=str, help="file containing which molecular descriptors to keep")
args = parser.parse_args()


def read_input_files():
    start_time = time.clock()
    # create an empty list to store the dataframes of protein features
    df_pro_list = []
    protein_features = pd.DataFrame(columns=["proteinName", "cluster_number"])
    drug_features = pd.DataFrame()
    protein_drug_features = pd.DataFrame(columns=["proteinName", "cluster_number"])

    # for each input file of protein features, load the dataframe then append to the dataframe list
    if args.p is not None:
        for path in args.p:
            df = parse_file(path)
            df_pro_list.append(df)

        protein_features = reduce(lambda x, y: pd.merge(x, y, on=["proteinName", "cluster_number"]), df_pro_list)
    del df_pro_list

    if args.pm is not None:
        protein_drug_features = parse_file(args.pm)

        # merge protein-molecular features with protein features
        protein_drug_features = pd.merge(protein_drug_features, protein_features, how="left",
                                         on=['proteinName', "cluster_number"])
        del protein_features

    else:
        protein_drug_features = protein_features
        del protein_features

    if args.m is not None:
        drug_features = parse_file(args.m)

        # do a pairwise merge (inner join) of molecular features with all protein-molecular features
        protein_drug_features = pd.merge(protein_drug_features, drug_features, on="moleculeName")

    del drug_features

    labels_df = protein_drug_features[["proteinName", "moleculeName", "label"]]

    protein_drug_features.drop(["label"], axis=1, inplace=True)
    protein_drug_features = pd.merge(protein_drug_features, labels_df)
    protein_drug_features.drop(["proteinName", "moleculeName"], axis=1, inplace=True)
    protein_drug_features = protein_drug_features.replace(to_replace=("", " "), value=np.nan)
    protein_drug_features.columns = [x.replace("/", "_") for x in protein_drug_features.columns]
    save_to_hdf5(protein_drug_features)
    protein_drug_features.to_csv('data/ml_pro_features_labels.csv', index=False, header=True)
    protein_drug_metadata = open("data/ml_pro_features_labels_metadata.txt", "w")
    protein_drug_metadata.write(str(list(protein_drug_features)))
    protein_drug_metadata.close()

    print("Output files generated in ", str(time.clock() - start_time), " seconds.")


def save_to_hdf5(data_frame):
    output_file = h5py.File("data/ml_pro_features_labels.h5", "w", libver='latest')
    data_frame.convert_objects(convert_numeric=True, inplace=True)

    for feature in data_frame:
        output_file.create_dataset(str(feature), [data_frame.shape[0], 1], data=data_frame[feature].tolist())
        data_frame.drop([feature], axis=1, inplace=True)

    output_file.close()


def parse_file(filepath):
    '''
        Given a filepath, determines which parser to run in order to extract data. Returns a pandas DataFrame.
        :param filepath: path to the input file
        :return: pandas DataFrame
        '''

    data = pd.DataFrame()

    if filepath.find('docking') != -1:
        data = load_protein_molecular_features(filepath)
    elif filepath.find('MolecularDescriptors') != -1:
        data = load_molecular_descriptors(filepath, args.feats)
    elif filepath.find('protein_features') != -1:  # protein_features_coach_avg and protein_features_2struc
        data = load_protein_features(filepath)
    else:
        print('File not supported')

    return data


# create a function for reading each input file type

def load_molecular_descriptors(filepath, descriptorsListFile=None):
    '''
        reads input files containing molecular descriptors
        :param filepath: path to the input file
        :return: pandas DataFrame
        '''

    data = pd.read_csv(filepath, delimiter='\t', low_memory=False)
    data.drop(["No."], axis=1, inplace=True)

    # rename duplicated moleculars
    for index, row in data.iterrows():
        molName = data.get_value(index, 'NAME', takeable=False)
        if molName.find('_') == -1:
            duplicatedMol = data.loc[data['NAME'] == molName]
            rowNum = len(duplicatedMol.index)

            # there are duplicated Mol.
            # rename from second molecule _#
            molIndex = 0
            if (rowNum > 1):
                for innerIndex, innerRow in duplicatedMol.iterrows():
                    molIndex = molIndex + 1
                    if (molIndex > 1):
                        newMolName = molName + '_' + str(molIndex)
                        data.set_value(innerIndex, 'NAME', newMolName)

    # select descriptorsList if there is
    if descriptorsListFile != None:
        with open(descriptorsListFile) as f:
            descriptorsList = f.read().splitlines()

        descriptorsList.append('NAME')

        for index, column in data.iteritems():  # index =  column name
            if index not in descriptorsList:
                data.drop(index, axis=1, inplace=True)

    # rename the second column to use it as key in the merge
    data.rename(columns={'NAME': 'moleculeName'}, inplace=True)

    # data.drop(["No."], axis=1, inplace=True)

    # convert keys to lowercase to prevent confusion
    data['moleculeName'] = data['moleculeName'].apply(lambda x: x.lower())

    return data


def load_protein_molecular_features(filepath):
    data = pd.read_csv(filepath)

    # extract protein name from Ligand and store it in new column
    data['proteinName'] = data['Filename'].str.extract('(...._cluster\d+)', expand=True)

    # extract molecule name from Ligand and store it in new column
    data['moleculeName'] = data['Filename'].str.extract('((?<=cluster\d_)\w+)', expand=True)

    # colunms' name to keep
    columns = ['proteinName', 'moleculeName', 'dockingEnergy', 'mmgbsaEnergy', 'avg_gauss1', 'avg_gauss2',
               'avg_repulsion', 'avg_hydrophobic', 'avg_hydrogen', 'Model1.gauss1', 'Model1.gauss2', 'Model1.repulsion',
               'Model1.hydrophobic', 'Model1.hydrogen', 'label']

    data = data[columns]

    # convert elements in the key column to lowercase to prevent confusion
    data['proteinName'] = data['proteinName'].apply(lambda x: x.lower())
    data['moleculeName'] = data['moleculeName'].apply(lambda x: x.lower())

    # create column for cluster number
    cluster_numbers = data["proteinName"].replace(to_replace='.*_cluster', value="", regex=True)
    data.insert(2, 'cluster_number', value=cluster_numbers)

    return data


def load_protein_features(filepath):
    '''
        reads protein_features_2struc
        :param filepath: path to the input file
        :return: pandas DataFrame
        '''
    data = pd.read_csv(filepath, delimiter=',')

    # rename the first column to use it as key in the merge

    if (list(data.columns.values))[0] == 'Cluster_Name':  # protein_features_2struc.csv
        data.rename(columns={'Cluster_Name': 'proteinName'}, inplace=True)
    elif (list(data.columns.values))[0] == 'cluster_name':  # protein_features_coach_avg.csv
        data.rename(columns={'cluster_name': 'proteinName'}, inplace=True)

    # convert elements in the key column to lowercase to prevent confusion
    data['proteinName'] = data['proteinName'].apply(lambda x: x.lower())

    # create column for cluster number
    cluster_numbers = data["proteinName"].replace(to_replace='.*_cluster', value="", regex=True)
    data.insert(1, 'cluster_number', value=cluster_numbers)

    return data


read_input_files()
