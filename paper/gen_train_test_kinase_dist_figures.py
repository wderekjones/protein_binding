import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import seaborn as sns
import argparse
import glob

parser = argparse.ArgumentParser(description="generate figures for class frequency distribution")

parser.add_argument("--train",type=str,help="path to train data")
parser.add_argument("--test",type=str,help="path to test data")

args = parser.parse_args()

sns.set_context("paper", rc={"font.size":4,"axes.titlesize":10,"axes.labelsize":8})   



def make_frequency_plot(df,title,output_path):
	plt.clf()
	sns.countplot(x="0",data=df)
	plt.title(title)
	plt.ylabel('Frequency')
	plt.xlabel('Kinase')
	plt.xticks(fontsize = 8, rotation='vertical') # work on current fig

	plt.savefig(output_path)

if args.train is not None:
	X_train_df = pd.read_csv(args.train)
	make_frequency_plot(X_train_df,"Training Set Kinase Frequency Distribution","x_train_compound_frequency.png")
if args.test is not None:
	X_test_df = pd.read_csv(args.test)
	make_frequency_plot(X_test_df,"Testing Set Kinase Frequency Distribution","x_test_compound_frequency.png")


