import tensorflow as tf
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
from keras.layers import SpatialDropout1D
from keras import objectives
from keras import backend as K


def l1Loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def shape(depth, row, col):
    if(K.image_dim_ordering() == 'th'):
        return (depth, row, col)
    else:
        return (row, col, depth)

def decoder_fc(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8, model_complexity = 1):
    pool_size = 2  
    layers = int(np.log2(img_dim_col) - 3)
    mod_input = Input(shape=shape(h,init_dim_row,init_dim_col))
    x = Conv2D(int(n*model_complexity), (11, 5), padding="same")(mod_input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.5)(x)
    x = Conv2D(n, (11, 5), padding="same")(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.5)(x)

    for i in range(layers):
        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(int(n*model_complexity), (11, 5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(.5)(x)
        
    x = Conv2D(channels, (11, 5), padding="same")(x)
    x = Activation("elu", name ="ae_out")(x)
    image_output = Lambda(lambda l: l + 1)(x)
    return Model(mod_input,image_output)

def encoder_fc(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8):
    layers = int(np.log2(img_dim_col) - 2)
    mod_input = Input(shape=shape(channels,img_dim_row, img_dim_col))

    x = Conv2D(n, (11, 5), padding="same")(mod_input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.5)(x)

    for i in range(1, layers):
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(n*i, (11, 5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(.5)(x)
    
    x = Conv2D(h, (11, 5), padding="same")(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.5)(x)
   
    return Model(mod_input,x)


def classifier_fc(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8, model_complexity = 1):

    mod_input = Input(shape=shape(h,init_dim_row,init_dim_col))
    # x = Dense(n*init_dim_col*init_dim_row)(mod_input)
    # x = Reshape(shape(n, init_dim_row, init_dim_col))(x)
    z = Conv2D(int(h*model_complexity),(9,4), padding = "valid")(mod_input)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.5)(z)
    z = Conv2D(int(h*model_complexity),(9,3), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.5)(z)
    z = Conv2D(int(h*model_complexity),(9,3), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.5)(z)
    z = Dense(2)(z)
    z = Reshape((2,), input_shape = (1,1,2,))(z)
    class_output = Activation("softmax", name = "class_out")(z)
    model = Model(inputs = [mod_input], outputs = [class_output])
    return model

def classifier_dilation(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8, model_complexity = 1):
    mod_input = Input(shape=shape(h,init_dim_row,init_dim_col))
    z = Conv2D(int(h*model_complexity),(9,5), dilation_rate = (2,1),padding = "valid")(mod_input)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.5)(z)
    z = Conv2D(int(h*model_complexity),(9,4), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.5)(z)
    z = Dense(2)(z)
    z = Reshape((2,), input_shape = (1,1,2,))(z)
    class_output = Activation("softmax", name = "class_out")(z)
    model = Model(inputs = [mod_input], outputs = [class_output])
    return model

def classifier_dilation2(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8, model_complexity = 1):
    mod_input = Input(shape=shape(h,init_dim_row,init_dim_col))
    z = Conv2D(int(h*model_complexity),(8,4), dilation_rate = (3,2),padding = "valid")(mod_input)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.75)(z)
    z = Conv2D(int(h*model_complexity),(4,2), padding = "valid")(z)
    z = BatchNormalization()(z)
    z = ELU()(z)
    z = Dropout(.75)(z)
    z = Dense(2)(z)
    z = Reshape((2,), input_shape = (1,1,2,))(z)
    class_output = Activation("softmax", name = "class_out")(z)
    model = Model(inputs = [mod_input], outputs = [class_output])
    return model

def segmenter_fc(h,img_dim_row, img_dim_col, channels, n = 128, init_dim_row = 25, init_dim_col = 8):

    model_complexity = .5
    pool_size = 2       
    mod_input = Input(shape=shape(h,init_dim_row,init_dim_col))
    x = Conv2D(int(h*model_complexity), (11,5), padding='same')(mod_input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.75)(x)
    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int((h/2)*model_complexity), (11,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.75)(x)
    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int((h/4)*model_complexity), (11,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.75)(x)
    x = UpSampling2D(size=(pool_size,pool_size))(x)
    x = Conv2D(int((h/8)*model_complexity), (11,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(.75)(x)
    x = Conv2D(2, (1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Reshape((64*200,2))(x)
    pixel_output = Activation('softmax', name ="seg_out")(x)
    model = Model(inputs = [mod_input], outputs = [pixel_output])
    return model

def ss_mask( encoder, decoder, classifier, segmenter, img_dim_row = 200, img_dim_col = 64, channels = 1):
    mod_input = Input(shape=shape(channels,img_dim_row, img_dim_col))
    x = encoder(mod_input)
    class_output = classifier(x) ; class_output = Lambda(lambda x: x, name='class_out')(class_output)
    seg_output = segmenter(x) ; seg_output = Lambda(lambda x: x, name='seg_out')(seg_output)
    decoded_output = decoder(x)  ; decoded_output = Lambda(lambda x: x, name='ae_out')(decoded_output)
    model = Model(inputs = [mod_input], outputs = [class_output, seg_output, decoded_output])
    return model

def s_mask( encoder,  classifier, segmenter, img_dim_row = 200, img_dim_col = 64, channels = 1):
    mod_input = Input(shape=shape(channels,img_dim_row, img_dim_col))
    x = encoder(mod_input)
    class_output = classifier(x) ; class_output = Lambda(lambda x: x, name='class_out')(class_output)
    seg_output = segmenter(x) ; seg_output = Lambda(lambda x: x, name='seg_out')(seg_output)
    model = Model(inputs = [mod_input], outputs = [class_output, seg_output])
    return model
