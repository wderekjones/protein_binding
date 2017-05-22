from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import objectives
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.preprocessing import normalize
from utils import *
#from plot_checkpoint import Plot_Reduction
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')

time_stamp = float(time.clock())


X_,y_ = load_data_h5("data/ml_pro_features_labels.h5")
X_ = normalize(X_)

num_epochs = 1
encoding_dim = 2
learning_rate = 1e-4

input_data = Input(shape=(189,))

encoded = BatchNormalization()(input_data)
encoded = Dense(100)(encoded)
encoded = PReLU()(encoded)
encoded = Dense(50)(encoded)
encoded = PReLU()(encoded)
encoded = Dense(25)(encoded)
encoded = PReLU()(encoded)
encoded = Dense(encoding_dim)(encoded)
encoded = PReLU()(encoded)

decoded = Dense(25)(encoded)
decoded = PReLU()(decoded)
decoded = Dense(50)(decoded)
decoded = PReLU()(decoded)
decoded = Dense(100)(decoded)
decoded = PReLU()(decoded)
decoded = Dense(189)(decoded)
decoded = PReLU()(decoded)

autoencoder = Model(input_data, decoded)
print (autoencoder.summary())

encoder = Model(input_data, encoded)
print (encoder.summary())

encoded_input = Input(shape=(encoding_dim,))


decoder = Model(encoded_input,autoencoder.layers[-1](autoencoder.layers[-2](autoencoder.layers[-3](autoencoder.layers[-4](
    autoencoder.layers[-5](autoencoder.layers[-6](autoencoder.layers[-7](autoencoder.layers[-8](encoded_input)))))))))

print (decoder.summary())


autoencoder.compile(optimizer=optimizers.rmsprop(lr=learning_rate),loss=objectives.mean_squared_error)


autoencoder.fit(X_,X_,epochs=num_epochs,batch_size=64,shuffle=True,validation_split=0.2,callbacks=[ModelCheckpoint(str(time_stamp)+"_model.h5")])

reduced_x = encoder.predict(X_)
plt.clf()
plt.scatter(reduced_x[:,0],reduced_x[:,1],c = y_,s=10)
plt.savefig(str(time_stamp)+"_final_dim_reduction.png")
