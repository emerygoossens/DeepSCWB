import numpy as np
import sys
sys.path.append("/code/")
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
import cv2
from keras import objectives
from keras import backend as K


from ae_helpers import *
from ae_models import *



data_points = 208104 - 10000
rand_ind = np.random.choice(range(data_points), size = data_points)
X_train = np.load("../data/X_train_sc.npy")[rand_ind,]
Y_train_seg = np.load("../data/Y_seg_train_sc.npy")[rand_ind,]
Y_train_class = np.load("../data/Y_class_train_sc.npy")[rand_ind,]
Y_train_image = X_train


good_ind = np.where(Y_train_class[:,1] == 1)[0]
num_bad = len(np.where(Y_train_class[:,0] == 1)[0])
num_good = len(good_ind)

good_weight_class = num_bad/float(num_good)

num_good_seg = Y_train_seg[good_ind,:,0].sum()
num_bad_seg = len(good_ind)*200*64 - num_good_seg

good_weight_seg = num_bad_seg/float(num_good_seg)

good_weight_seg_class_ratio = good_weight_seg/float(good_weight_class)

experiment = str(sys.argv[1])


if experiment == "seg_class_only":
    percent_labeled = 1.
    image_sample_weight = np.zeros(X_train.shape[0]) + 10e-18
elif experiment == "ten_percent":
    percent_labeled = .1
    image_sample_weight = np.ones(X_train.shape[0])
elif experiment == "one_percent":
    percent_labeled = .01
    image_sample_weight = np.ones(X_train.shape[0])

#zero out weights to simulated unlabeled_data
class_sample_weight = np.zeros(X_train.shape[0]) + 10e-18
nonzero_weight_ind = np.random.choice(range(class_sample_weight.shape[0]), int(class_sample_weight.shape[0]*percent_labeled), replace = False)
class_sample_weight[nonzero_weight_ind] = 1



seg_sample_weight = np.copy(class_sample_weight)
is_bad = np.where(Y_train_class[:,0] == 1)[0]
seg_sample_weight[is_bad] = 0 
seg_sample_weight = seg_sample_weight/percent_labeled
class_sample_weight = class_sample_weight/percent_labeled




model_name, model_path, estp, mchp, rlrp = model_id_init(es_patience = 20, rl_patience = 5, rl_factor = .5)



nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam_opt  = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


batch_size = 64
epochs = 1000




def weighted_pixelwise_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([good_weight_seg,1],"float32")),-1)
    return - tf.reduce_mean(class_weight_loss)

def weighted_class_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([1*good_weight_seg_class_ratio,good_weight_class*good_weight_seg_class_ratio],"float32")),-1)
    return - tf.reduce_mean(class_weight_loss)



print(model_name)
print(model_path)



if experiment == "seg_class_only":
    model = seg_class(model_complexity = .25)
    model.compile(loss_weights = [1,1],loss=[weighted_pixelwise_crossentropy, weighted_class_crossentropy],optimizer=adam_opt, metrics=["accuracy"] )
    print(experiment)
    model.fit(x = X_train, y = [Y_train_seg, Y_train_class],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            sample_weight = {"pixel_output":seg_sample_weight, "class_output":class_sample_weight},
            validation_split = .05,
            callbacks=[estp, mchp, rlrp],
            verbose=2)
else:
    model = semisup_seg_class(model_complexity = .25)
    model.compile(loss_weights = [1,1,percent_labeled],loss=[weighted_pixelwise_crossentropy, weighted_class_crossentropy, "mae"],optimizer=adam_opt, metrics=["accuracy"] )
    print(experiment)
    model.fit(x = X_train, y = [Y_train_seg, Y_train_class, Y_train_image],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            sample_weight = {"pixel_output":seg_sample_weight, "class_output":class_sample_weight, "image_output":image_sample_weight},
            validation_split = .05,
            callbacks=[estp, mchp, rlrp],
            verbose=2)




print(model_name)
print(model_path)



# y_pred = model.predict(X_train[0:1])
# y_pred_seg_test = y_pred[0][:,:,0]
# y_pred_seg_reshape = np.reshape(y_pred_seg_test,(200,64))
# cv2.imwrite("images/seg_good_img_pred_test.png", y_pred_seg_reshape*255)




