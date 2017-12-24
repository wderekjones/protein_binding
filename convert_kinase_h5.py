# this script converts the old h5 file for the kinase subset, which contains no encoded train/test split, to an h5
# with the explicitly encoded train/test split
# by Derek Jones on 12/19/2017

import argparse
import h5py
import pandas as pd
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('-i',type=str,help="input file path to old h5")
parser.add_argument('-o',type=str,help="output file path to new h5")
parser.add_argument('-train', type=str, help="input path to file with training compounds")
parser.add_argument('-test', type=str, help="input path to file with testing compounds")
args = parser.parse_args()


old_fo = h5py.File(args.i, "r")
new_fo = h5py.File(args.o, "w")

receptors = list(old_fo.keys()) #extract the receptor names which were stored as root level keys....dumb
features = list(old_fo[receptors[0]].keys()) #extract the features, just use the keys from the first receptor all are same

train_df = pd.read_csv(args.train,index_col=[0]) #expect the first column to be the idxs
test_df = pd.read_csv(args.test,index_col=[0]) #expect the first column to be the idxs

# I know this code is terrible so please save the sarcastic comments...I'm doing my rest rn @ 12:53am

# create the train split
new_fo.create_group("train")
for receptor in receptors:
    new_fo['train'].create_group(receptor)
    # build up a data_frame or numpy array with all of the feature information for a given receptor, then store these as datasets in the new fo
    receptor_df = train_df[train_df['0'] == receptor]
    drugIDs = receptor_df['1']
    temp_df_list = []

    for feature in features:
        feature_list = []
        for drug in drugIDs:
            idx, _ = np.where(np.asarray(old_fo[receptor+"/drugID"]) == drug)
            feature_i = old_fo[str(receptor)+"/"+str(feature)][int(idx)]
            feature_list.append(feature_i)
        new_fo['train'+'/'+str(receptor)+'/'+str(feature)] = np.asarray(feature_list)

# create the train split
new_fo.create_group("test")
for receptor in receptors:
    new_fo['test'].create_group(receptor)
    # build up a data_frame or numpy array with all of the feature information for a given receptor, then store these as datasets in the new fo
    receptor_df = test_df[test_df['0'] == receptor]
    drugIDs = receptor_df['1']
    temp_df_list = []

    for feature in features:
        feature_list = []
        for drug in drugIDs:
            idx,_ = np.where(np.asarray(old_fo[receptor+"/drugID"]) == drug)
            print(idx)
            feature_i = old_fo[str(receptor)+"/"+str(feature)][int(idx)]
            feature_list.append(feature_i)
        new_fo['test'+'/'+str(receptor)+'/'+str(feature)] = np.asarray(feature_list)

new_fo.close()
old_fo.close()