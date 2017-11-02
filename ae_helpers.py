from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
import os
from scipy import signal

import random
import tensorflow as tf
import random


def model_id_init(es_patience = 50, rl_patience = 10, rl_factor = .9, monitor_es = "val_loss"):
    model_name = ''.join(random.choice('0123456789ABCDEF') for i in range(6))
    ensure_dir("../saved_models/")
    model_path = "../saved_models/model_{}/".format(model_name)
    ensure_dir(model_path)
    estp = EarlyStopping(monitor=monitor_es,patience=es_patience,verbose=2)
    mchp = ModelCheckpoint("{}/model_{}".format(model_path, model_name), save_best_only = True)
    rlrp = ReduceLROnPlateau(factor = rl_factor,patience=rl_patience, min_lr=.00001)
    print("Model Info:")
    print(model_name)
    print(model_path)
    return model_name, model_path, estp, mchp, rlrp



def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def reformat(data):
	dataset = data.reshape(
	(-1, 200, 50, 1)).astype(np.float32)
	return dataset


def computeIoU(y_pred_batch, y_true_batch):
    return tf.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 

def pixelAccuracy(y_pred, y_true):
    y_pred = np.argmax(np.reshape(y_pred,[y_pred.shape[0],2,200,64]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[y_true.shape[0],2,200,64]),axis=0)
    y_pred = y_pred * (y_true>0)
    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)



def smooth(x,window_len=20,window='flat'):
    #print(len(s))
    w=np.ones([window_len,window_len],'d')
    y = signal.convolve2d(x, w/w.sum(), boundary='symm', mode='same')
    return y




def _loss_tensor(y_true, y_pred):
    weight = weights
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred)  + (1- y_true) * K.log(1-y_pred))
    weight_map = np.ones((y_pred.shape[0],200*64,), dtype=np.float32)
    weight_map[y_true[:,0,:]==1] = weight
    weighted_out = 1.0*out
    return K.mean(out, axis = -1)


def weighted_pixelwise_crossentropy(class_weights):
    
    def loss(y_true, y_pred):
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss



# def iou(y_true, y_pred):
#     # true positive / (true positive + false positive + false negative)
    

