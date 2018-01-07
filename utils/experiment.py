import h5py
import pickle
import itertools
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from utils.input_pipeline import load_data


def get_feature_list(data_path):
    input_fo = h5py.File(data_path, "r")
    feature_list = list(input_fo.keys())
    feature_list.remove("label")
    return feature_list


def pickle_model(title, clf):
    output_pickle = open(("results/" + str(title) + ".pkl"), "wb")
    pickle.dump(clf, output_pickle)
    output_pickle.close()


def output_model_parameters(title, clf, y_test, preds):
    output_file = open("results/" + title + ".txt", "w")
    output_file.write(title + "\n")
    output_file.write(classification_report(y_test, preds, target_names=["Class 0", "Class 1"], digits=4))
    output_file.write("\n" + clf.__str__())
    output_file.close()


def output_classifier_confusion(title, confusion):
    plt.clf()
    plot_confusion_matrix(confusion, classes=[0, 1], title=title)
    plt.tight_layout()
    plt.savefig("results/"+ title + "_confusion.png")


def output_classifier_roc(title, y_test, scores):
    fpr, tpr, _ = roc_curve(y_test, scores[:, 1])
    auc = roc_auc_score(y_test, scores[:, 1])

    bench_fpr, bench_tpr, _ = roc_curve(y_test,np.zeros(y_test.shape[0]))

    plt.clf()
    plt.plot(fpr, tpr, label=title)
    plt.plot(bench_fpr, bench_tpr, label = "Total Negative Bias")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(str(title) + "\n AUC: " + str(auc))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("results/" + title + "_roc.png",  bbox_extra_artists=(lgd,), bbox_inches='tight')


def save_results(title, clf, X_test, y_test):
    '''

    :param report_title: a string that will be used as the report filename
    :param clf: a scikit learn classifier object
    :param preds: the predictions generated from the classifier
    :param y_test: the counterpart ground truth values
    :return: nothing, outputs a file and generates corresponding confusion matrix
    '''

    # get the predictions, their probabilities, and classifier confusion
    preds = clf.predict(X_test)
    scores = clf.predict_proba(X_test)
    confusion = confusion_matrix(y_test, preds)

    # pickle the model
    pickle_model(title, clf)

    # generate .txt containing precision/recall/f1-score and hyperparameters
    output_model_parameters(title, clf, y_test, preds)

    # generate the output confusion matrix
    output_classifier_confusion(title, confusion)

    # generate the output roc curve
    output_classifier_roc(title, y_test, scores)


def optimize_hyper_param(clf_mode, title, X_, y_, random_seed, n_iterations):
    clf = None
    best_h = 1
    best_f1_score = 0
    f1_scores = []
    train_times = []
    skf = StratifiedKFold(n_splits=n_iterations, random_state=random_seed)

    i = j = 1
    for train_index, test_index in skf.split(X_, y_.flatten()):
        X_train_strat, X_test_strat = X_[train_index], X_[test_index]
        y_train_strat, y_test_strat = y_[train_index], y_[test_index]

        if clf_mode == "knn":
            clf = KNeighborsClassifier(n_neighbors=j, n_jobs=-1)
        elif clf_mode == "rf":
            clf = RandomForestClassifier(n_estimators=pow(2, j), max_depth=3, n_jobs=-1, random_state=random_seed)
        elif clf_mode == "ada":
            weak_learner = DecisionTreeClassifier(max_depth=3, random_state=random_seed)
            clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=j,random_state=random_seed)

        t_i = time.clock()
        clf.fit(X_train_strat, y_train_strat.flatten())
        t_j = time.clock()
        train_times.append((t_j - t_i))
        clf_preds = clf.predict(X_test_strat)
        clf_f1 = f1_score(y_test_strat.flatten(), clf_preds.flatten())
        if clf_f1 > best_f1_score:
            best_f1_score = clf_f1
            best_h = j

        f1_scores.append(clf_f1)
        j += 1

    if clf_mode == "knn":
        clf = KNeighborsClassifier(n_neighbors=best_h)
        print("best n_neighbors: ", best_h)
    elif clf_mode == "rf":
        clf = RandomForestClassifier(n_estimators=best_h, max_depth=3, n_jobs=-1, random_state=random_seed)
        print("best n_estimators: ", best_h)
    elif clf_mode == "ada":
        weak_learner = DecisionTreeClassifier(max_depth=3, random_state=random_seed)
        clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=best_h, random_state=random_seed)
        print("best n_estimators: ", best_h)

    plt.clf()
    f, (subp1, subp2) = plt.subplots(2, sharex=True)
    if clf_mode == "knn":
        subp1.set_title(title + " Training Time vs. $k$")
        subp2.set_title(title + " Test F1 Score vs. $k$")
        subp2.set_xlabel("$k$")
    if clf_mode == "rf" or clf_mode == "ada":
        subp1.set_title(title + " Training Time vs. n_estimators")
        subp2.set_title(title + " Test F1 Score vs . n_estimators")
        subp2.set_xlabel("n_estimators")
    subp1.plot(list(range(i, j, 1)), train_times)
    subp1.set_ylabel("seconds")
    subp2.plot(list(range(i, j, 1)), f1_scores)
    subp2.set_ylabel("F1")

    subp2.set_xticks(list(range(i, j, 2)))
    plt.savefig("results/" + title + ".png")

    return clf


def get_average_performance(clf, X_, y_, random_seed, n_iterations):
    f1_scores = []
    skf = StratifiedKFold(n_splits=n_iterations, random_state=random_seed)
    i = j = 1
    for train_index, test_index in skf.split(X_, y_.flatten()):
        X_train_strat, X_test_strat = X_[train_index], X_[test_index]
        y_train_strat, y_test_strat = y_[train_index], y_[test_index]

        clf.fit(X_train_strat, y_train_strat.flatten())
        clf_preds = clf.predict(X_test_strat)
        f1 = f1_score(y_test_strat.flatten(), clf_preds.flatten())

        f1_scores.append(f1)
        j += 1

    average_f1 = np.mean(f1_scores)

    return average_f1


def run_full_experiment(clf_mode, title, X_, y_, random_seed, n_iterations):
    t0 = time.clock()
    clf = optimize_hyper_param(clf_mode, title, X_, y_, random_seed, n_iterations)
    t1 = time.clock()
    avg_f1 = get_average_performance(clf, X_, y_, random_seed, n_iterations)
    print(title, "training complete in ", (t1-t0), " seconds.")

    return clf, avg_f1


def plot_feature_importance_curve(plot_title, plot_path, feature_support):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,8))
    indices = range(0,len(feature_support))
    sorted_importances = np.sort(feature_support)[::-1]
    benchmark = (1/len(feature_support))*np.ones([len(feature_support)])
    zeros = np.zeros([len(feature_support)])
    ax.plot(indices, sorted_importances, label="importances")
    ax.plot(indices, benchmark, label="$1/n$")
    ax.fill_between(indices, zeros, benchmark, where=benchmark > zeros,
                    alpha=0.5,interpolate=True)
    ax.set_title(plot_title)
    ax.set_ylabel("Feature Importance")
    fig.savefig(plot_path)
    plt.close()


def plot_roc_curve(plot_title, plot_path, clf_fpr, clf_tpr, clf_label, roc_score):
    plt.clf()
    plt.figure(figsize=[12, 8])
    plt.plot(clf_fpr, clf_tpr, lw=2, color='g', label=clf_label)

    plt.plot([0, 1], [0, 1], 'r--', lw=2, color='k')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(plot_title+": ROC = "+str(roc_score))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path,  bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_dummy_roc_comparison(plot_title, plot_path,clf,clf_label,X_train,y_train, X_test,y_test,random_state=None):
    # it may be better to include all dummy comparison code in one function...class?
    clf_tpr, clf_fpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

    random_strat_clf = DummyClassifier(strategy="stratified", random_state=random_state).fit(X_train,y_train)
    random_uniform_clf = DummyClassifier(strategy="uniform", random_state=random_state).fit(X_train,y_train)
    random_most_freq_clf = DummyClassifier(strategy="most_frequent", random_state=random_state).fit(X_train,y_train)

    random_strat_tpr, random_strat_fpr, _ = roc_curve(y_test,random_strat_clf.predict_proba(X_test))
    random_uniform_tpr, random_uniform_fpr, _ = roc_curve(y_test,random_uniform_clf.predict_proba(X_test))
    random_most_freq_tpr, random_most_freq_fpr, _ = roc_curve(y_test,random_most_freq_clf.predict_proba(X_test))

    plt.clf()
    plt.figure(figsize=[12, 8])
    plt.plot(clf_fpr, clf_tpr, lw=2, color='g', label=clf_label)
    plt.plot(random_most_freq_fpr, random_most_freq_tpr, lw=2, color='b', label="most frequent")
    plt.plot(random_strat_fpr, random_strat_tpr, lw=2, color='r', label="stratified")
    plt.plot(random_uniform_fpr, random_uniform_tpr, lw=2, color='y', label="uniform")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(plot_title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(plot_path, plot_title, cm, classes,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    plt.figure(figsize=[10, 8])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,decimals=2)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(plot_title)
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
    plt.savefig(plot_path)
    plt.close()


def output_classification_results(results_title, results_path, y_true, preds):
    result_file = open(results_path, "w")
    report = classification_report(y_true, preds)
    result_file.write(results_title+"\n" + str(report))
    result_file.close()
