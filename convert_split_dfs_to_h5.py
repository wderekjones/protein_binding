'''
The purpose of this script is to take the intermediate dataframes that contain the training/testing splits from the
paper "Polypharmacology Within the Full Kinome: a Machine Learning Approach" and create an h5 file with the explicit splits
by: Derek Jones
'''

import argparse
import pandas as pd
import numpy as np
import h5py
import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("-train")
parser.add_argument("-test")
parser.add_argument("-o")
args = parser.parse_args()

train_df = pd.read_csv(args.train)
test_df = pd.read_csv(args.test)
out_fo = h5py.File(args.o,"w")

out_fo.create_group("train")
out_fo.create_group("test")

receptor_list = train_df.receptor.unique().tolist()

print("creating training split")
start = time.clock()
for receptor in tqdm.tqdm(receptor_list):
    out_fo["train"].create_group(receptor)
    receptor_df = train_df[train_df.receptor == receptor]
    for feature in feature_list:
        out_fo["train/"+receptor+"/"+feature] = np.array(receptor_df[feature])

print("creating testing split")
for receptor in tqdm.tqdm(receptor_list):
    out_fo["test"].create_group(receptor)
    receptor_df = test_df[test_df.receptor == receptor]
    for feature in feature_list:
        out_fo["test/"+receptor+"/"+feature] = np.array(receptor_df[feature])
end = time.clock()
print("created h5 file in {} seconds".format(end-start))