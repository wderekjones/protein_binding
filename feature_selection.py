import time
import argparse
import h5py
import itertools
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
from sklearn.metrics import roc_curve,roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import uniform
from scipy.stats.distributions import uniform_gen
from scipy.stats import randint as sp_randint
from utils.input_pipeline import load_data
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

parser = argparse.ArgumentParser(description="iterative feature selection script")

parser.add_argument("--f",type=str,help="path to set of initial features")
parser.add_argument("--data",type=str,help="path to initial dataset, this should at least contain the features specified by --f if given")
args = parser.parse_args()

#
#           PUT ALL OF THIS INTO A LOOP, a good idea would be to remove all calls to functions that output to files
#               in order to debug efficiently without generating a bunch of trash.......
#

random_state=np.random.RandomState(0)


feature_path = args.f

imputer = Imputer()
normalizer = Normalizer()

data_path = args.data

with open(feature_path, "r") as input_file:
    feature_list = []
    for line in input_file:
        line = line.strip('\n')
        feature_list.append(line)

print(len(feature_list))



delta_error = 9e10
converged = False
prev_f1 = 0
cur_f1 = 0
while len(feature_list) > 0:
    X, y = load_data(data_path, features_list=feature_list)

    X_train,X_test,y_train,y_test = train_test_split(normalizer.fit_transform(imputer.fit_transform(X.astype(np.float32))),y.astype(np.float32),
                                                     stratify=y.astype(np.float32),test_size=0.2, shuffle=True,
                                                     random_state=random_state)
    del X
    del y

    rforest = RandomForestClassifier(n_jobs=10,oob_score=True, class_weight='balanced', criterion='GINI', random_state=random_state)
    forest_params = {"n_estimators": sp_randint(15,30),
                     "min_samples_leaf": sp_randint(1,100)
                   }

    forest_estimator = RandomizedSearchCV(rforest,forest_params,scoring='f1', random_state=random_state)
    forest_estimator.fit(X_train,y_train.flatten())
    best_forest = forest_estimator.best_estimator_

    best_forest_preds = best_forest.predict(X_test)

    cur_f1 = f1_score(y_test.flatten(),best_forest_preds)
    cur_acc = accuracy_score(y_test.flatten(),best_forest_preds)
    print("accuracy:",cur_acc,"\tf1-score",cur_f1)

    if (np.abs(prev_f1 - cur_f1) > 1e-3):
        full_features = feature_list
        support = best_forest.feature_importances_
        keep_idxs = support > np.mean(support,axis=0)

        features_to_keep = pd.DataFrame(np.asarray(full_features)[keep_idxs])
        feature_list = list(features_to_keep.values)
        #features_to_keep.to_csv("results/step2_features.csv",index=False,header=False)
    prev_f1 = cur_f1



def plot_feature_importance_curve():

    plt.clf()
    plt.figure(figsize=[12,8])
    plt.plot(np.sort(support)[::-1])
    plt.title("Step 1 Random Forest Feature Support (sorted)")
    plt.ylabel("feature importance")
    plt.savefig("poster_results/feature_importance_curve_step1.png")
    plt.show()


best_forest_fpr, best_forest_tpr, _ = roc_curve(y_test,best_forest.predict_proba(X_test)[:,1])

def plot_roc_curve():
    plt.clf()
    plt.figure(figsize=[12,8])
    plt.plot(best_forest_fpr, best_forest_tpr,lw=2, label=("RF: AUC = "+
                                                      str(np.round(roc_auc_score(y_test.flatten(),best_forest.predict(X_test),
                                                                                 average='weighted'),3))), color = 'g')
    plt.plot(log_reg_fpr, log_reg_tpr,lw=2, label=("Logistic Regression: AUC = "+
                                              str(np.round(roc_auc_score(y_test.flatten(),log_reg.predict(X_test),
                                                                         average='weighted'),3))), color = 'b')
    plt.plot([0,1], [0,1], 'r--',lw=2, label="Random: AUC = 0.5", color = 'k')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Comparison of Classifiers: Step 1 Random Forest Features")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("poster_results/classifier_comparison_step1.png",  bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



    rforest_confusion = confusion_matrix(y_test, best_forest_preds)
    plt.clf()
    plt.figure(figsize=[10, 8])
    plot_confusion_matrix(rforest_confusion, classes=[0, 1], title="Random Forest Step 1 Confusion")
    plt.tight_layout()
    plt.savefig("poster_results/random_forest_step1_confusion.png")
    plt.show()

def output_classification_results():
    result_file = open("poster_results/step1_test_results.txt", "w")
    best_forest_report = classification_report(y_test, best_forest_preds)

    print("Random Forest Test Set Performance\n", best_forest_report)
    result_file.write(str("Step 1 Random Forest Test Set Performance\n" + str(best_forest_report)))

    result_file.close()