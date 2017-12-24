''' this script takes the full dataset, along with either the testing or training compounds used in the paper
"Polypharmacology Within the Full Kinome: a Machine Learning Approach" and creates a dataframe containing the data
for each of the splits. This is used to then create an h5 file with the training/testing splits
by: Derek Jones
'''

import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i',type=str,help="input path to full dataset")
parser.add_argument('-split', type=str, help="input path to the file containing training or testing compounds (i.e. the split)")
parser.add_argument('-o', type=str, help="output path")
args = parser.parse_args()


full_df = pd.read_csv(args.i)
compounds = pd.read_csv(args.split)
split_df = pd.DataFrame()

receptor_list = full_df.receptor.unique().tolist()

for receptor in tqdm(receptor_list):
    receptor_drugs = compounds[compounds['0'] == receptor]['1'].tolist()
    receptor_data_list = []
    receptor_df = full_df[full_df["receptor"] == receptor]
    for drug in receptor_drugs:
        receptor_row = receptor_df[receptor_df.drugID == drug]
        receptor_data_list.append(receptor_row)
    split_df = pd.concat([split_df,pd.concat(receptor_data_list)])

split_df.to_csv(args.o,index=False)
