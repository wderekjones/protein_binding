import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import seaborn as sns
import argparse
import glob

parser = argparse.ArgumentParser(description="generate value counts for dataset")

parser.add_argument("--data",type=str,help="path to data")
parser.add_argument("--o",type=str,help="output data path")
args = parser.parse_args()

if args.data is not None:
	df = pd.read_csv(args.data,index_col=0)
	
	kinases = np.unique(df["0"]).tolist()
	count_df = pd.DataFrame()
	for kinase in kinases:
		kinase_value_counts = df[df["0"] == kinase]["label"].value_counts()
		count_df[kinase] = kinase_value_counts.as_matrix()
	count_df = count_df.transpose()
	count_df.to_csv(args.o)
