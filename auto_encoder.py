import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import normalize
from utils import *



X_,y_ = load_data_h5("data/ml_pro_features_labels.h5")

mb_size = 64
z_dim = 6
X_dim = X_.shape[1]
y_dim = y_.shape[1]
h_dim = 50
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
kl_loss = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(labels=z_mu, logits = z_logvar))

# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0


X_n,_ = load_data_h5("data/ml_pro_features_labels.h5",mode=0)
X_n = normalize(X_n)
X_p,_ = load_data_h5("data/ml_pro_features_labels.h5",mode=1)
X_p = normalize(X_p)
mb_sample_pos = np.random.choice(X_p.shape[0], int(mb_size / 2), replace=False)

for it in range(1000000):
    mb_sample_neg = np.random.choice(X_n.shape[0],int(mb_size/2),replace=False)
    X_mb = np.vstack((X_n[mb_sample_neg],X_p[mb_sample_pos]))
    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})
    kl_ = sess.run(kl_loss,feed_dict={X:X_mb})
    if it % 1000 == 0:
        print('Iter: {}'.format(it),'\t --- \t','Loss: {:.4}'. format(loss))#,'\t','-----','kl loss: ',np.mean(kl_,axis=0))

    if it % 100000 == 0:
        mb_sample_pos = np.random.choice(X_p.shape[0], int(mb_size / 2), replace=False)


'''from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras_tqdm import TQDMNotebookCallback
from utils import *


X,y = load_data_h5("data/ml_pro_features_labels.h5")


encoding_dim = 2
input_sample = Input(shape=(189,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_sample)
encoded = Dense(100,activation='sigmoid')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(100,activation='sigmoid')(encoded)
decoded = Dense(189, activation='sigmoid')(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_sample, decoded)

autoencoder.compile(optimizer=optimizers.RMSprop(),loss='mean_squared_error')
autoencoder.fit(X,X,epochs = 100,callbacks=[TQDMNotebookCallback()], validation_split=0.2)'''