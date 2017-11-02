

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

nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

adam_opt  = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def segnet(pool_size = 2):
    image_input = Input(shape = (200,50,1,))
    x = ZeroPadding2D((0,7))(image_input)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(128*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(256*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    encoded = ELU()(x)


    #Decoder

    x = Conv2D(int(256*model_complexity), (3,3), padding='same')(encoded)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(128*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)


    x = Conv2D(2, (1, 1), padding='valid')(x)

    x = Reshape((2,200,64,), input_shape=(200,64,2,))(x)
    # x = Permute((2, 1))(x)
    pixel_output = Activation('softmax', name ="seg_output")(x)
    model = Model(inputs = [image_input], outputs = [pixel_output])
    model.compile(loss=_loss_tensor,optimizer=nadam_opt, metrics=["accuracy"])
    model.summary()
    return model


def segnet_classifier():

    #Encoder
    image_input = Input(shape = (200,50,1,))
    x = ZeroPadding2D((0,7))(image_input)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(128*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(int(256*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    encoded = ELU()(x)



    z = Conv2D(int(256*model_complexity),(9,3), padding = "valid")(encoded)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.25)(z)
    z = Conv2D(int(256*model_complexity),(9,3), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.25)(z)
    z = Conv2D(int(256*model_complexity),(9,4), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.25)(z)
    z = Dense(2)(z)
    z = Reshape((2,), input_shape = (1,1,2,))(z)
    class_output = Activation("softmax", name = "class_output")(z)


    #Decoder

    x = Conv2D(int(256*model_complexity), (3,3), padding='same')(encoded)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(128*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int(64*model_complexity), (3,3), padding='same')(x)
    x = BatchNormalization()(x)


    x = Conv2D(2, (1, 1), padding='valid')(x)

    x = Reshape((2,200,64,), input_shape=(200,64,2,))(x)
    # x = Permute((2, 1))(x)
    pixel_output = Activation('softmax', name ="seg_output")(x)
    model = Model(inputs = [image_input], outputs = [pixel_output, class_output])
    model.compile(loss=_loss_tensor,optimizer=nadam_opt, metrics=["accuracy"])
    model.summary()
    return model


