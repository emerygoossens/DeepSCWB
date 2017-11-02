import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
import random
# from hparams import hparams
from keras.layers import SpatialDropout1D


from keras import objectives
from keras import backend as K

from ae_helpers import *
from ae_models import *



model_name =   ''.join(random.choice('0123456789ABCDEF') for i in range(6))
ensure_dir("saved_models/")
model_path = "saved_models/model_{}/".format(model_name)
ensure_dir(model_path)






batch_size = 128
image_size_1 = 200
image_size_2 = 50
num_channels = 1
epochs = 1000


estp = EarlyStopping(monitor='val_loss',patience=50,verbose=2)
mchp = ModelCheckpoint("{}/model_{}".format(model_path, model_name), save_best_only = True)
rlrp = ReduceLROnPlateau(factor = .9,patience=10, min_lr=.0000001)




X_train = np.load("data/X_train_segnet.npy")
Y_train = np.load("data/Y_train_segnet.npy")


weights = Y_train[:,1,:,:].sum()/Y_train[:,0,:,:].sum()




nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam_opt  = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



model = segnet2()

model.compile(loss=[pixelwise_crossentropy],optimizer=adam_opt, metrics=["accuracy",computeIoU])






model.fit(x = X_train, y = [Y_train],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split = .05,
        callbacks=[estp, mchp, rlrp],
        verbose=2)
