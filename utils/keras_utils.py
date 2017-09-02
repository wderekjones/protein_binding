import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from utils.experiment import plot_confusion_matrix


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def generate_report_keras(report_title, clf, X_test, y_test):
    '''

    :param report_title: a string that will be used as the report filename
    :param clf: a scikit learn classifier object
    :param preds: the predictions generated from the classifier
    :param y_test: the counterpart ground truth values
    :return: nothing, outputs a file and generates corresponding confusion matrix
    '''

    preds = clf.predict(X_test)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    # generate .txt containing precision/recall/f1-score and hyperparameters
    output_file = open("results/" + report_title + ".txt", "w")
    output_file.write(report_title + "\n")
    output_file.write(classification_report(y_test, preds, target_names=["Class 0", "Class 1"], digits=4))
    output_file.write("\n" + clf.__str__())
    output_file.close()

    # generate the output confusion matrix
    confusion = confusion_matrix(y_test, preds)
    plt.clf()
    plot_confusion_matrix(confusion, classes=[0, 1], title=report_title)
    plt.tight_layout()
    plt.savefig("results/" + report_title + "_confusion.png")

    # generate the output roc curve
    scores = clf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(str(report_title) + "\n AUC: " + str(auc))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("results/" + report_title + "_roc.png")
