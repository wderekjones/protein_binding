import time
import argparse
import h5py
import os
import arrow
import itertools
from utils.input_pipeline import output_feature_summary
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import uniform
from scipy.stats.distributions import uniform_gen
from scipy.stats import randint as sp_randint
from utils.input_pipeline import load_data, read_feature_list, get_feature_histogram
from utils.experiment import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support

# TODO: fix plotting to look less terrible and instead, ready for publication....http://www.jesshamrick.com/2016/04/13/reproducible-plots/
# TODO: try standardizing the data
# TODO: find a better way to give unique names to each of the output directories for the feature selection loop
# TODO: run this algorithm on each of the subsets of features, then combine these features and run feature selection on that final set
# TODO: move as many function definitions to the utils.experiment module as possible to avoid redundancy and keep the script short
# TODO: output optimal hyperparameters
# TODO: implement dummy classifier to generate biased/random predictions

# Reminder: compare performance  on each feature set with and without imputation versus dropping all null features

parser = argparse.ArgumentParser(description="iterative feature selection script")

parser.add_argument("--f", type=str, help="path(s) to set of initial features")
parser.add_argument("--data", type=str, help="path to initial dataset, this should at least contain the features specified by --f if given")
parser.add_argument("--null", type=str, help="path to null features")
parser.add_argument("--strat", type=str, help="imputation strategy used to fill in the null values")
parser.add_argument("--out", type=str, help="output path to dir")
args = parser.parse_args()

random_state = np.random.RandomState(0)

current_output_directory = arrow.now().format('YYYY-MM-DD_HH:mm:ss')
def run_iterative_forest_selection(feature_path, output_directory=None, null_path=None, undersampled=None):
    if output_directory is None:
        output_directory = current_output_directory
    else:
        output_directory = output_directory+current_output_directory+"/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    best_metric_sum = 0


    data_path = args.data
    n_nulls = 0
    with open(feature_path, "r") as input_file:
        feature_list = []
        for line in input_file:
            line = line.strip('\n')
            feature_list.append(line)
    print("")
    print("Number of features before removing nulls "+str(len(feature_list)))
    if null_path is not None:
        with open(null_path, "r") as null_input_file:
            for line in null_input_file:
                n_nulls += 1
                line = line.strip('\n')
                if line in feature_list:
                    feature_list.remove(line)
    print("")
    print("Number of features after removing "+str(n_nulls)+" nulls "+str(len(feature_list)))

    forest_params = {"n_estimators": sp_randint(30, 100),
                     "min_samples_leaf": sp_randint(1, 100),
                     "max_features": ["sqrt", "log2"]
                     }

    full_features = pd.DataFrame(feature_list)
    best_features_to_keep = pd.DataFrame(full_features)
    step = 0
    max_steps = 10
    while len(full_features) > 0 and step < max_steps:

        # Load the data
        print("|full_features| = ",full_features.shape[0])
        X, y = load_data(data_path, features_list=list(full_features[0]))
        if args.null is not None:
            X = Imputer(strategy=args.strat).fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), y.astype(np.float32),
                                stratify=y.astype(np.float32), test_size=0.2, shuffle=True, random_state=random_state)

        # Instantiate and train the model. Optimize hyperparameters by sampling from integer distributions
        rforest = RandomForestClassifier(n_jobs=12, oob_score=True, class_weight='balanced', bootstrap=True,
                                         criterion='gini', random_state=random_state)

        # find the optimal setting of parameters that maximizes the weighted f1 measure, using 100 samples of
        # 5-fold cross validation. Then select the best forest and predict on the testing set.
        forest_estimator = RandomizedSearchCV(rforest, forest_params, cv=3, scoring='f1_weighted', n_jobs=5, random_state=random_state)
        forest_estimator.fit(X_train, y_train.flatten())
        best_forest = forest_estimator.best_estimator_
        best_forest_preds = best_forest.predict(X_test)

        # compute the false positive and true positive rates for the positive class (confusing)
        best_forest_fpr, best_forest_tpr, _ = roc_curve(y_test, best_forest.predict_proba(X_test)[:, 1])

        # extract the feature importances, visualize, and then save them as the set of features for the next iteration
        support = best_forest.feature_importances_
        keep_idxs = support > np.mean(support, axis=0)
        features_to_keep = pd.DataFrame(np.asarray(list(full_features[0]))[keep_idxs])
        print("features_to_keep.shape = ",features_to_keep.shape)
        if len(features_to_keep.values) < 1:
            break
        else:
            #features_to_keep.to_csv(output_directory + "step" + str(step)+"_most_important_feats" + ".csv", index=False, header=False)
            full_features.to_csv(output_directory + "step" + str(step) + ".csv",
                                    index=False, header=False)


        # plot the ROC curves of the set of features that the forest was trained upon
        plot_roc_curve("Step " + str(step) + " ROC", output_directory + "step" + str(step) + "_roc", best_forest_fpr,
                       best_forest_tpr, clf_label="Random Forest")

        # compute and plot the confusion matrices for the set of features that the forest was trained upon
        rforest_confusion = confusion_matrix(y_test, best_forest_preds)
        plot_confusion_matrix(cm=rforest_confusion, classes=[0, 1], plot_title="Random Forest Step "+str(step)+" Confusion",
                              plot_path=output_directory+"step"+str(step)+"_confusion.png")

        # compute the importances of the features that the forest was trained upon
        plot_feature_importance_curve("Step " + str(step) + " feature importances",
                                          output_directory+"step" + str(step) + "_feature_importances", support)

        # output a summary of the classification performance broken down by class
        output_classification_results("Step "+str(step)+" classification report\n", output_directory+"step"+
                                          str(step)+"_classification_report.txt", y_test, best_forest_preds)

        # output a breakdown of feature proportions per class.
        # TODO: visualize these as a histogram over the 4 possible categories
        output_feature_summary(output_directory+"step"+str(step)+"feature_summary.txt", "data/all_kinase/with_pocket/full_kinase_set_features_list.csv",
                               output_directory+"step"+str(step)+".csv")

        feature_histogram = get_feature_histogram(list(full_features[0]), protein_feature_path="data/all_kinase/with_pocket/protein_features_list.csv",
                              pocket_feature_path="data/all_kinase/with_pocket/pocket_features_list.csv",
                              drug_feature_path="data/all_kinase/with_pocket/drug_features_list.csv",
                              binding_feature_path="data/all_kinase/with_pocket/binding_features_list.csv")

        plt.clf()
        sns.countplot(x="feature_class", data=feature_histogram)

        plt.savefig(output_directory+"step"+str(step)+"_countplot.png")

        # compute the precision, recall, and fscore of the predicitions

        # extract the values for the undersampled class, in this case the positive class
        if undersampled is not None:
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test.flatten(), best_forest_preds,
                                                                           average='weighted', labels=[undersampled])
        else:
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test.flatten(), best_forest_preds)

        if len(support) > 0:
            if (np.sum([precision, recall, fscore])/len(support)) > best_metric_sum:
                best_metric_sum = np.sum([precision, recall, fscore])/len(support)
                best_features_to_keep = full_features

        step += 1

        #full_features = list(features_to_keep[0])
        full_features = features_to_keep

    return best_features_to_keep

best_feature_set = set()
for i in range(10):
    best_features = run_iterative_forest_selection(args.f, args.out, args.null, undersampled=1)
    best_feature_set.union(set(best_features[0]))


pd.DataFrame(list(best_feature_set)).to_csv(args.out+current_output_directory+"/best_result.csv")
