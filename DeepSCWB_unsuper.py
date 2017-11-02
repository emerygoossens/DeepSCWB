import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import *
from keras.models import Model
from keras.layers import Input,Lambda
from keras.layers.core import Flatten, Dense, Reshape, Permute
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, Conv2DTranspose,MaxPooling2D, ZeroPadding2D
from keras.layers import Input,merge, Lambda
from keras.backend import relu
from keras.activations import softmax
from keras.optimizers import Nadam, Adam
from keras.layers import LSTM, concatenate
from keras.layers.embeddings import Embedding
import h5py
import os
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import SpatialDropout1D

from keras import objectives
from keras import backend as K

from code.ae_helpers import *
from code.ae_models import *


X_train = np.load("data/XY_train_unsuper.npy")
Y_train = X_train


model_name, model_path, estp, mchp, rlrp = model_id_init(es_patience = 50, rl_patience = 10, rl_factor = .9)



nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam_opt  = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



batch_size = 512
epochs = 1000



model = autoencoder()
model.compile(loss="mean_squared_error",optimizer=adam_opt, metrics=["mae"])
model.fit(x = X_train, y = [Y_train],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split = .05,
        callbacks=[estp, mchp, rlrp],
        verbose=2)
