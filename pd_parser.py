import argparse
import pandas as pd
import h5py
from functools import reduce


#TODO: add metadata output

parser = argparse.ArgumentParser(description="Process files containing protein binding affinity features")

parser.add_argument('--p', type=str, nargs='+', help="list paths of files containing protein features")

parser.add_argument('--pm', type=str, help="list paths of files containing protein-molecular compound features")

parser.add_argument('--m', type=str, help="file containing molecular descriptors")

parser.add_argument('--feats', type=str, help="file containing which molecular descriptors to keep")
args = parser.parse_args()


def read_input_files():
    # create an empty list to store the dataframes of protein features
    df_pro_list = []

    # for each input file of protein features, load the dataframe then append to the dataframe list
    for path in args.p:
        df = parse_file(path)
        df_pro_list.append(df)

    df_agg_pro = reduce(lambda x, y: pd.merge(x, y, on=["proteinName"]), df_pro_list)
    pro_drug_df = parse_file(args.pm)

    # merge protein-molecular features with protein features
    pro_drug_all_df = pd.merge(pro_drug_df, df_agg_pro, how='left', on='proteinName')
    mol_df = parse_file(args.m)

    # do a pairwise merge (inner join) of molecular features with all protein-molecular features
    output_df = pd.merge(pro_drug_all_df, mol_df, on="moleculeName")
    labels_df = output_df[["proteinName", "moleculeName", "label"]]

    # drop the labels from the features dataframe
    output_df.drop(["label"], axis=1, inplace=True)
    output_df = pd.merge(output_df, labels_df)
    output_df.drop(["proteinName", "moleculeName"], axis=1, inplace=True)
    save_to_hdf5(output_df)
    output_df.to_csv('data/ml_pro_features_labels.csv', index=False, header=False)
    output_df_metadata = open("data/ml_pro_features_labels.txt","w")
    output_df_metadata.write(str(list(output_df)))
    output_df_metadata.close()


def save_to_hdf5(data_frame):
    output_file = h5py.File("data/ml_pro_features_labels.h5", "w")
    data_frame = data_frame.convert_objects(convert_numeric=True)

    for feature in data_frame:
        feature_list = data_frame[feature].tolist()
        output_file.create_dataset(str(feature), [data_frame.shape[0], 1], data=feature_list)

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
