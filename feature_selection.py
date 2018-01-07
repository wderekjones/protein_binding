import argparse
import os
import arrow
from utils.input_pipeline import output_feature_summary
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import seaborn as sns
import numpy as np
from scipy.stats import randint as sp_randint
from utils.input_pipeline import get_feature_histogram
from utils.experiment import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description="iterative feature selection script")

parser.add_argument("-f", type=str, help="path(s) to set of initial features")
parser.add_argument("-data", type=str, help="path to initial dataset, this should at least contain the features specified by --f if given")
parser.add_argument("-null", type=str, help="path to null features")
parser.add_argument("-strat", type=str, help="imputation strategy used to fill in the null values")
parser.add_argument("-label",type=str,help="optional specify target label")
parser.add_argument("-out", type=str, help="output path to dir")
parser.add_argument("-prot", type=str, help="a flag that indicates that protein names are used as labels")
parser.add_argument("-names", type=str,nargs='+', help="list of proteins to exclude from training", default=None)
parser.add_argument("-root", type=str, help="root path for data and feature lists")
parser.add_argument("--split", type=str, help="if using dataset with seperate train/test splits, specify the split", default=None)
args = parser.parse_args()

random_state = 0
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

        # Write the features used for this iteration to a file
        full_features.to_csv(output_directory + "step" + str(step) + ".csv", index=False, header=False)

        # Load the data
        data = load_data(data_path, protein_name_list=args.names, features_list=list(full_features[0]), label=args.label, split=args.split)
        print("Step {}: training using {} features.".format(step,full_features.shape[0]))


        X_train, X_test, y_train, y_test = train_test_split(data, data["label"].as_matrix(),
                        stratify=data["label"].as_matrix(), test_size=0.2, shuffle=True, random_state=random_state)

        X_train[["receptor","drugID","label"]].to_csv(output_directory+"/step "+str(step)+"_training_compounds.csv",index=False)
        X_train = X_train[X_train.columns.difference(["receptor", "drugID", "label"])]

        X_test[["receptor","drugID","label"]].to_csv(output_directory+"/step "+str(step)+"_testing_compounds.csv",index=False)
        X_test = X_test[X_test.columns.difference(["receptor", "drugID", "label"])]


        # Instantiate and train the model. Optimize hyperparameters by sampling from integer distributions
        rforest = RandomForestClassifier(n_jobs=-1, oob_score=True, class_weight='balanced', bootstrap=True,
                                         criterion='gini', random_state=random_state)
        imputer = Imputer(strategy=args.strat)

        # find the optimal setting of parameters that maximizes the weighted f1 measure, using 100 samples of
        # 3-fold cross validation. Then select the best forest and predict on the testing set.
        forest_estimator = RandomizedSearchCV(rforest, forest_params, cv=3, scoring='f1_weighted', n_jobs=5, random_state=random_state)
        forest_estimator.fit(imputer.fit_transform(X_train.as_matrix()), y_train.flatten())
        best_forest = forest_estimator.best_estimator_
        best_forest_preds = best_forest.predict(imputer.fit_transform(X_test))

        #TODO: export the predictions..add these as a column in the X_test dataframe?

        # extract the feature importances, visualize, and then save them as the set of features for the next iteration
        support = best_forest.feature_importances_


        # compute the false positive and true positive rates for the positive class (confusing)
        best_forest_fpr, best_forest_tpr, _ = roc_curve(y_test, best_forest.predict_proba(imputer.fit_transform(X_test))[:,1])

        # plot the ROC curves of the set of features that the forest was trained upon
        plot_roc_curve("Step " + str(step) + " ROC", output_directory + "step" + str(step) + "_roc", best_forest_fpr,
                       best_forest_tpr, clf_label="Random Forest", roc_score=roc_auc_score(y_test,best_forest.predict_proba(imputer.fit_transform(X_test))[:,1]))

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
        output_feature_summary(output_directory+"step"+str(step)+"feature_summary.txt", args.root+"/full_kinase_set_features_list.csv",
                               output_directory+"step"+str(step)+".csv")

        feature_histogram = get_feature_histogram(list(full_features[0]), protein_feature_path=args.root+"/protein_features_list.csv",
                              pocket_feature_path=args.root+"/pocket_features_list.csv",
                              drug_feature_path=args.root+"/drug_features_list.csv",
                              binding_feature_path=args.root+"/binding_features_list.csv")

        # TODO: Normalize the values in the histogram
        sns.countplot(x="feature_class", data=feature_histogram)

        plt.savefig(output_directory+"step"+str(step)+"_countplot.png")

        #compute the precision, recall, and fscore of the predicitions extract the values for the undersampled class
        precision,recall,fscore = 0,0,0
        if undersampled is not None:
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test.flatten(), best_forest_preds,
                                                                           labels=[undersampled])
        else:
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test.flatten(), best_forest_preds, average='macro')

        keep_idxs = support > np.mean(support, axis=0)
        features_to_keep = pd.DataFrame(np.asarray(list(full_features[0]))[keep_idxs])


        if len(support) > 0:
            # find the mean of the precision, recall, and fscore. Divide this by the number of features, compare
            # to the largest value computed so far
            #if (np.sum([precision, recall, fscore])/3)/(len(support)) > best_metric_sum:
            # so incorporating the magnitude of the feature set results in a selection that is much too aggressive
            if np.sum([precision, recall, fscore])/3 >= best_metric_sum:
                best_metric_sum = np.sum([precision, recall, fscore])/3
                best_features_to_keep = full_features

        step += 1
        print("Step {}:  Precision: {} \t Recall: {} \t F1: {}".format(step, precision, recall, fscore))
        print("Step {}:  Reduced from {} to {} features".format(step, full_features.shape[0], features_to_keep.shape[0]))
        full_features = features_to_keep

    return best_features_to_keep


best_features = run_iterative_forest_selection(args.f, args.out, args.null)
best_features.to_csv(args.out+current_output_directory+"/best_result.csv")
