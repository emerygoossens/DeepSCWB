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


data_points = 50000
rand_ind = np.random.choice(range(200000), size = data_points)
X_train = np.load("data/X_train_sc.npy")[rand_ind,]
Y_train_seg = np.load("data/Y_seg_train_sc.npy")[rand_ind,]
Y_train_class = np.load("data/Y_class_train_sc.npy")[rand_ind,]








model_name, model_path, estp, mchp, rlrp = model_id_init(es_patience = 50, rl_patience = 10, rl_factor = .9)



nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam_opt  = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

200*64

batch_size = 64
epochs = 1000

x_true = X_train[:10]
y_true_class = Y_train_class[:10]
y_true_seg = Y_train_seg[:10]



def pixelwise_crossentropy(y_true, y_pred):
    # y_true = y_true_seg
    # y_pred = y_pred_seg
    weights = np.ones((y_pred.shape[0]))
    is_bad = np.where(y_true[:,:,0].sum(-1) == 0)[0]
    loss_multiplier = tf.constant(10e-8 + y_true.shape[0]/(10e-8 + y_true.shape[0] - len(is_bad)),"float32" )
    weights[is_bad] = 0
    weights = tf.cast(weights,"float32")
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    weighted_loss = tf.multiply(y_true * tf.log(y_pred), tf.constant([4.,1.],"float32"))
    sum_weighted_loss = tf.reduce_sum(weighted_loss,[1,2])
    zeroed_weighted_loss= tf.multiply(weights, sum_weighted_loss)
    return -tf.reduce_mean(weighted_loss)
    # return - tf.multiply(tf.reduce_mean(zeroed_weighted_loss), loss_multiplier) - tf.constant(10e-8,"float32")

# def pixelwise_crossentropy(target, output):
#     output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
#     return - tf.reduce_sum(target * tf.log(output))



def weighted_pixelwise_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    return - tf.reduce_mean(tf.multiply(y_true * tf.log(y_pred), tf.constant([3.,.75],"float32")))

def weighted_class_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    return - tf.reduce_mean(tf.multiply(y_true * tf.log(y_pred), tf.constant([.5,4.5],"float32")))


def weighted_pixelwise_crossentropy2(y_true, y_pred):
    weights = tf.ones((y_pred.shape[0],))
    is_bad = tf.where(y_true[:,:,0].sum(-1) == 0)
    loss_multiplier = tf.constant(10e-8 + y_true.shape[0]/(10e-8 + y_true.shape[0] - len(is_bad)),"float32" )
    # weights[is_bad] = 0
    weights = tf.cast(weights,"float32")
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    weighted_loss = tf.multiply(y_true * tf.log(y_pred), tf.constant([4.,1.],"float32"))
    weighted_loss = tf.reduce_sum(weighted_loss,-1)
    weighted_loss = tf.reduce_sum(weighted_loss,-1)
    # weighted_loss = tf.multiply(weighted_loss,weights)
    return - tf.reduce_sum(weighted_loss)






# def _loss_tensor(y_true_seg, y_pred_seg):
# y_pred_seg = K.clip(y_pred_seg, 10e-8, 1.0- 10e-8)
# out = -(y_true_seg * K.log(y_pred_seg)  + (1- y_true_seg) * K.log(1-y_pred_seg))
# weight_map = np.ones(shape = (y_pred_seg.shape[0],200*64))
# weight_map[y_true_seg[:,0]==1] = 4
# is_bad = np.where(y_true_seg[:,:,0].sum(-1) == 0)[0]
# weight_map[is_bad] = 0
# weighted_out = weight_map*out
#     return K.mean(out, axis = -1)

    # def loss(y_true, y_pred):
    #     epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    #     y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    #     return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

seg_sample_weight = np.ones(Y_train_class.shape[0])
is_bad = np.where(Y_train_class[:,0] == 1)[0]
seg_sample_weight[is_bad] = 0 
model = segnet_classifier(model_complexity = .25)

model.compile(loss=[weighted_pixelwise_crossentropy, weighted_class_crossentropy],optimizer=adam_opt, metrics=["accuracy"] )


model.fit(x = X_train, y = [Y_train_seg, Y_train_class],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        sample_weight = {"pixel_output":seg_sample_weight},
        validation_split = .05,
        callbacks=[estp, mchp, rlrp],
        verbose=2)


y_pred_seg, y_pred_class = model.predict(x_true)

y_true_seg[:,:,0].sum(-1)
is_bad = np.where(y_true_seg[:,:,0].sum(-1) == 0)[0]


- tf.reduce_sum(tf.multiply(y_true_class * tf.log(y_pred_class), tf.constant([1., .25])),-1)


def class_weighted_pixel_loss(y_true_seg,y_pred_seg):
    weights = np.ones(y_pred_seg.shape[0])
    is_bad = np.where(y_true_seg[:,:,0].sum(-1) == 0)[0]
    weights[is_bad] = 0
    weights = K.cast(weights,"float32")
    loss = -(y_true_seg * K.log(y_pred_seg)  + (1- y_true_seg) * K.log(1-y_pred_seg))
    class_weights = np.ones((2,1))
    class_weights[0] = 4
    class_weights = K.cast(class_weights, "float32")
    class_weighted_loss = K.squeeze(K.dot(loss, class_weights),-1)
    return K.mean(class_weighted_loss)




y_pred = K.clip(y_pred, 10e-8, 1. - 10e-8)
loss = - K.multiply(y_true * K.log(y_pred)), tf.constant([1., .25])



















