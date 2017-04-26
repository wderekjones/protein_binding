from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import *

# need to make sure that each class is included

X_p, y_p = get_positive_data("data/ml_pro_features_labels.csv", 3000)

avg_svm_accuracy = 0
avg_forest_accuracy = 0
avg_logreg_accuracy = 0

avg_svm_f1 = 0
avg_forest_f1 = 0
avg_logreg_f1 = 0


for i in range(0,10):
    X_n, y_n = get_negative_data("data/ml_pro_features_labels.csv", 3000)
    X = combine_positive_negative_data(X_n, X_p)
    y = combine_positive_negative_data(y_n, y_p)
    imputer = Imputer()
    X = imputer.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    rforest = RandomForestClassifier(n_estimators=10)
    rforest.fit(X_train, y_train)
    forest_preds = rforest.predict(X_test)
    forest_confusion = confusion_matrix(y_test, forest_preds, labels=[0, 1])
    plot_confusion_matrix(forest_confusion, classes=[0, 1], title="Random Forest "+str(i)+" Confusion")
    plt.savefig('results/forest_test_'+str(i)+'.png')
    forest_accuracy = accuracy_score(y_test, forest_preds)
    forest_f1 = f1_score(y_test,forest_preds)
    avg_forest_accuracy += forest_accuracy
    avg_forest_f1 += forest_f1
    print("Random Forest accuracy: ", forest_accuracy,"\t F1-score: ",forest_f1)


    svm = SVC()
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_confusion = confusion_matrix(y_test, svm_preds)
    plt.clf()
    plot_confusion_matrix(svm_confusion, classes=[0, 1], title="SVM "+str(i)+" Confusion")
    plt.savefig('results/svm_test_'+str(i)+'.png')
    svm_accuracy = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test,svm_preds)
    avg_svm_accuracy += svm_accuracy
    avg_svm_f1 += svm_f1
    print("SVM accuracy: ", svm_accuracy,"\t F1-score: ",svm_f1)

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    logistic_preds = logistic.predict(X_test)
    logistic_confusion = confusion_matrix(y_test, logistic_preds)
    plt.clf()
    plot_confusion_matrix(logistic_confusion, classes=[0, 1], title="Logistic Regression "+str(i)+" Confusion")
    plt.savefig('results/logistic_test_'+str(i)+'.png')
    logistic_accuracy = accuracy_score(y_test, logistic_preds)
    logistic_f1 = f1_score(y_test,logistic_preds)
    avg_logreg_accuracy += logistic_accuracy
    avg_logreg_f1 += logistic_f1
    print("Logistic Regression accuracy: ", logistic_accuracy,"\t F1-score: ",logistic_f1)

    plt.clf()

avg_forest_accuracy = avg_forest_accuracy/10.0
avg_forest_f1 = avg_forest_f1/10.0
avg_svm_accuracy = avg_svm_accuracy/10.0
avg_svm_f1 = avg_svm_f1/10.0
avg_logreg_accuracy = avg_logreg_accuracy/10.0
avg_logreg_f1 = avg_logreg_f1 / 10.0

print("--------------------------------------------------------------\n")

print ("Random Forest Average Accuracy: ", avg_forest_accuracy,"\t F1-score: ",avg_forest_f1)
print ("SVM Average Accuracy: ", avg_svm_accuracy,"\t F1-score: ",avg_svm_f1)
print ("Logistic Regression Average Accuracy: ", avg_logreg_accuracy,"\t F1-score: ",avg_logreg_f1)