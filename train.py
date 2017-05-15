import tensorflow as tf
import keras.backend as K
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from utils import *

# sess = tf.Session()

# init = tf.group(tf.local_variables_initializer(),tf.global_variables_initializer())
# sess.run(init)


#TODO: create a new file for the keras code/utilities
#TODO: add functionality to save the model weights




K.tf.set_random_seed(0)


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


X_p, y_p = load_data("data/ml_pro_features_labels.csv", 4000, mode=1)

X_n, y_n = load_data("data/ml_pro_features_labels.csv", 100000, mode=0)
X = combine_positive_negative_data(X_n, X_p)
y = combine_positive_negative_data(y_n, y_p)

# y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Sequential()

model.add(Dense(189, input_shape=(189,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=[precision, recall])

print(model.summary())

model.fit(X_train, y_train, batch_size=1000, epochs=100, validation_split=0.25, callbacks=[EarlyStopping()])
print("Evaluating on test data: \n")
# metrics = model.evaluate(X_test, y_test)

report_title = "MLP"

preds = model.predict(X_test)

preds[preds >= 0.5] = 1
preds[preds < 0.5] = 0

generate_report("MLP", model, preds, y_test)
