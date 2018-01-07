'''
The purpose of this script is to take a csv file containing protein_binding data and create an h5 with a specified proportion
of examples to hold as a testing set.
by: Derek Jones
'''
import time
import os
import h5py
import pandas as pd
import numpy as np
import argparse
import numpy as np
from tqdm import tqdm
random_state = np.random.RandomState(0)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="path to old csv file")
parser.add_argument("-o", type=str, help="path to new h5 file")
parser.add_argument("-c", type=str, help="prefix for compound lists")
parser.add_argument("--a", type=float, help="ratio of examples to hold out for test set", default=0.2) #create 80/20 split
args = parser.parse_args()


def save_to_hdf5(data_frame, output_name):

    output_file = h5py.File(output_name, "w", libver='latest')
    column_names = list(data_frame.columns.values)
    group_names = list(set(data_frame['receptor']))
    output_file.create_group("train")
    output_file.create_group("test")


    for group_name in tqdm(group_names):
        grp_data = data_frame[data_frame['receptor'] == group_name]
        labels = data_frame[data_frame['receptor'] == group_name]["label"]
        idxs = np.arange(0, len(labels))
        train_idxs, test_idxs = train_test_split(idxs, stratify=labels, test_size=args.a, random_state=random_state)
        data_frame.iloc[train_idxs][["receptor","drugID","label","vina_score"]].to_csv(args.c+"_training_compounds.csv")
        data_frame.iloc[test_idxs][["receptor", "drugID", "label", "vina_score"]].to_csv(args.c + "_testing_compounds.csv")
        output_file['/train'].create_group(group_name)
        output_file['/test'].create_group(group_name)
        for feature in iter(column_names):
            # print(feature)
            if feature == "label":
                output_file['train/'+str(group_name)].require_dataset(str(feature), [grp_data.iloc[train_idxs].shape[0], 1],
                                             data=np.asarray(grp_data.iloc[train_idxs][feature]),
                                                dtype=np.int8)
                output_file['test/'+str(group_name)].require_dataset(str(feature), [grp_data.iloc[test_idxs].shape[0], 1], data=np.asarray(grp_data.iloc[test_idxs][feature]),
                                             dtype=np.int8)
            elif feature in ['receptor','drugID','Filename']:

                output_file['train/'+str(group_name)].require_dataset(str(feature), [grp_data.iloc[train_idxs].shape[0], 1],
                                        data=np.asarray(grp_data.iloc[train_idxs][feature]),
                                        dtype=h5py.special_dtype(vlen=str))
                output_file['test/'+str(group_name)].require_dataset(str(feature), [grp_data.iloc[test_idxs].shape[0], 1],
                                            data=np.asarray(grp_data.iloc[test_idxs][feature]),
                                            dtype=h5py.special_dtype(vlen=str))
            else:
                output_file['train/'+str(group_name)+"/"+str(feature)] = np.asarray(pd.to_numeric(grp_data.iloc[train_idxs][feature]),dtype=np.float16)
                output_file['test/'+str(group_name)+"/"+str(feature)] = np.asarray(pd.to_numeric(grp_data.iloc[test_idxs][feature]),dtype=np.float16)

    output_file.close()




t0 = time.clock()
save_to_hdf5(pd.read_csv(args.i, keep_default_na=False,na_values=[np.nan, 'na']).convert_objects(convert_numeric=True), args.o)
t1 = time.clock()
print(args.i, "converted to .h5 in", (t1-t0), "seconds.")

