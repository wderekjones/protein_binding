import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from utils import *
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

random_state = 0
t0 = time.clock()
X_p, y_p = load_data_h5("data/ml_pro_features_labels.h5", mode=1)
X_n, y_n = load_data_h5("data/ml_pro_features_labels.h5", mode=0)

print("Data loaded in ", (time.clock() - t0), " seconds.")

X = combine_positive_negative_data(X_n, X_p)
y = combine_positive_negative_data(y_n, y_p)


X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Train-Test split completed in ", (time.clock() - t0), " seconds.")

#t0 = time.clock()
#manifold_train = Isomap(n_neighbors=1,n_components=10).fit_transform(X_train)
#manifold_test = Isomap(n_neighbors=1, n_components=10).fit_transform(X_test)
#print("Dimensionality reduction completed in ", (time.clock() - t0))

#rforest0 = RandomForestClassifier(random_state=random_state)
#t1 = time.clock()
#rforest0.fit(X_train, y_train)
#print ("Random forest trained on full features in ", (time.clock() - t1), " seconds.")
#rforest_preds0 = rforest0.predict(X_test)
#generate_report("Random Forest: Manifold Features", rforest0, rforest_preds0, y_test)

#rforest1 = RandomForestClassifier(random_state=random_state)
#t2 = time.clock()
#rforest1.fit(manifold_train, y_train)
#print("Random forest trained on manifold features in ",(time.clock() - t2), " seconds." )
#rforest_preds1 = rforest1.predict(manifold_test)
#generate_report("Random Forest: Manifold Features", rforest1, rforest_preds1, y_test)


#svm0 = SVC(random_state=random_state)
#t3 = time.clock()
#svm0.fit(X_train, y_train)
#print ("Support Vector Machine train on full features in ", (time.clock() - t3), " seconds.")
#svm_preds0 = svm0.predict(X_test)
#generate_report("SVM: Full Features", svm0, svm_preds0, y_test)

#svm1 = SVC(random_state=random_state)
#t4 = time.clock()
#svm1.fit(manifold_train, y_train)
#print ("Support Vector Machine trained on manifold features in ", (time.clock() - t4), " seconds.")
#svm_preds1 = svm1.predict(manifold_test)
#generate_report("SVM: Manifold Features", svm1, svm_preds1, y_test)


logistic0 = LogisticRegression(random_state=random_state)
t5 = time.clock()
logistic0.fit(X_train, y_train)
print ("Logistic Regression trained on full features in ", (time.clock() - t5), " seconds.")
logistic_preds0 = logistic0.predict(X_test)
generate_report("Logistic: Full Features", logistic0, logistic_preds0, y_test)

#logistic1 = LogisticRegression(random_state=random_state)
#t6 = time.clock()
#logistic1.fit(manifold_train, y_train)
#print ("Logistic Regression trianed on manifold features in ", time.clock() - t6)
#logistic_preds1 = logistic1.predict(manifold_test)
#generate_report("Logistic: Manifold Features", logistic1, logistic_preds1, y_test)


#----------------------------#

model0 = Sequential()

model0.add(Dense(188, input_shape=(188,), activation='sigmoid'))
model0.add(Dense(1, activation='sigmoid'))


model0.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=[precision, recall])

print(model0.summary())
t7 = time.clock()
model0.fit(X_train, y_train, batch_size=1000, epochs=100, validation_split=0.25, callbacks=[EarlyStopping()])
print ("MLP trained on full features in ", (time.clock() - t7), " seconds")

preds = model0.predict(X_test)

preds[preds >= 0.5] = 1
preds[preds < 0.5] = 0
generate_report("MLP: Full Features", model0, preds, y_test)

#---------------------------#

#model1 = Sequential()

#model1.add(Dense(manifold_train.shape[0], input_shape=(manifold_train.shape[0],), activation='sigmoid'))
#model1.add(Dense(1, activation='sigmoid'))


#model1.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=[precision, recall])

#print(model1.summary())
#t8 = time.clock()
#model1.fit(manifold_train, y_train, batch_size=1000, epochs=100, validation_split=0.25, callbacks=[EarlyStopping()])
#print ("MLP trained on manifold features in ", (time.clock() - t8), " seconds")


#preds = model1.predict(manifold_test)

#preds[preds >= 0.5] = 1
#preds[preds < 0.5] = 0
#generate_report("MLP: Manifold Features", model1, preds, y_test)