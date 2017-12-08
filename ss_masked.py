
import threading
import sys
sys.path.append("/code/")
import numpy as np
import keras.backend as K
from keras.preprocessing import image
from keras.datasets import cifar10
from keras.optimizers import Adam
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
import ss_models
import ss_trainer_all


# fname1 = "code/ss_main_mask.py"
# with open(fname1, 'r') as fin:
#     print fin.read()

# fname2 = "code/ss_trainer_all.py"
# with open(fname2, 'r') as fin:
#     print fin.read()

# fname3 = "code/ss_models.py"
# with open(fname3, 'r') as fin:
#     print fin.read()

experiment = str(sys.argv[1])

seed = int(sys.argv[2])
np.random.seed(seed = seed)

print("Running experiement: {}".format(experiment))


if experiment == "seg_class_all":
    data_percent = 1
    train_ae = False
    train_seg = True
    train_class = True
elif experiment == "seg_class_ae_10":
    data_percent = .1
    train_ae = True
    train_seg = True
    train_class = True
elif experiment == "class_seg_only_10":
    data_percent = .1
    train_ae = False
    train_seg = True
    train_class = True
elif experiment == "seg_class_ae_1":
    data_percent = .01
    train_ae = True
    train_seg = True
    train_class = True
elif experiment == "class_seg_only_1":
    data_percent = .01
    train_ae = False
    train_seg = True
    train_class = True
elif experiment == "seg_class_ae_all":
    data_percent = 1.
    train_ae = True
    train_seg = True
    train_class = True

X_train = np.load("data/X_train_ss.npy")
X_val = np.load("data/X_val_ss.npy")


Y_train_class = np.load("data/Y_train_class_ss.npy")
Y_val_class = np.load("data/Y_val_class_ss.npy")

Y_train_seg = np.load("data/Y_train_seg_ss.npy")
Y_val_seg = np.load("data/Y_val_seg_ss.npy")

X_train_seg = np.load("data/X_train_seg_ss.npy")
X_val_seg = np.load("data/X_val_seg_ss.npy")



X_train_class = np.load("data/X_train_class_ss.npy")
X_val_class = np.load("data/X_val_class_ss.npy")


pass_ind = np.where(Y_train_class[:,1] ==1)[0]
fail_ind = np.where(Y_train_class[:,1] ==0)[0]


ss_pass_num = int(pass_ind.shape[0]*data_percent)
ss_fail_num = int(fail_ind.shape[0]*data_percent)

rand_pass_ind = np.random.choice(pass_ind, size = ss_pass_num, replace = False)
rand_fail_ind = np.random.choice(fail_ind, size = ss_fail_num, replace = False)

class_rand_ind = np.union1d(rand_pass_ind, rand_fail_ind)

X_train_class = X_train_class[class_rand_ind]
Y_train_class = Y_train_class[class_rand_ind]

seg_rand_ind = np.argsort(rand_pass_ind)

X_train_seg = X_train_seg[seg_rand_ind]
Y_train_seg = Y_train_seg[seg_rand_ind]

good_ind = np.where(Y_train_class[:,1] == 1)[0]
num_bad = len(np.where(Y_train_class[:,0] == 1)[0])
num_good = len(good_ind)

good_weight_class = (num_bad/float(num_good))

num_good_seg = Y_train_seg[:,:,0].sum()
num_bad_seg = Y_train_seg[:,:,1].sum()

good_weight_seg = num_bad_seg/float(num_good_seg)

good_weight_seg_class_ratio = 1





epochs = 100
batches_per_epoch = (X_train.shape[0]/128)

#image parameters
img_size_row = 200 
img_size_col = 64
channels = 1 #1 for grayscale

#Model parameters
z = 1 #Generator input
n = 32
h = n*4 #Autoencoder hidden representation


nadam_opt_seg_class = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model_name, model_path, estp, mchp, rlrp= model_id_ss(experiment = experiment, seed = seed, es_patience = 20, rl_patience = 3, rl_factor = .5)


def mae_weighted(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)*.01


def weighted_pixelwise_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([good_weight_seg,1],"float32")),-1)
    return - tf.reduce_mean(class_weight_loss)

def weighted_class_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([1*good_weight_seg_class_ratio,good_weight_class*good_weight_seg_class_ratio],"float32")),-1)
    return - tf.reduce_mean(class_weight_loss)

def rand_unif_scale(img_data, lower = .8, upper = 1.2):
    unifs = np.random.uniform(lower, upper, img_data.shape[0])
    for j in range(img_data.shape[0]):
        img_data[j] = img_data[j]*unifs[j]
    return img_data

def rand_flip_input(img_data, prob = .5):
    for j in range(img_data.shape[0]):
        if np.random.binomial(1,prob):
            img_data[j] = np.flip(img_data[j],1)
        if np.random.binomial(1,prob):
            img_data[j] = np.flip(img_data[j],2)
    return img_data

def rand_flip_seg(seg_data_x, prob = .5):
    for j in range(seg_data_x.shape[0]):
        if np.random.binomial(1,prob):
            seg_data_x[j] = np.flip(seg_data_x[j],2)
    return seg_data_x

def rand_unif_scale_image(img_data, lower = .7, upper = 1.5):
    return img_data*np.random.uniform(lower, upper)


MASK_VALUE = [-1.,-1.]

def build_masked_loss(loss_function, mask_value = MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function

def class_acc(y_true, y_pred):
    total = K.sum( K.cast(K.not_equal(y_true, MASK_VALUE), K.floatx()))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    return correct / total

def seg_acc(y_true, y_pred):
    total = K.sum( K.cast(K.not_equal(y_true, MASK_VALUE), K.floatx()))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    return correct / total

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator


def gen_masked(aex, segx, segy, classx, classy, batch_size = 32, train_ae = True):
    seg_class_bs = batch_size*2
    if train_ae:
        ae_bs = batch_size*2
        seg_class_bs = batch_size
        ae_ymask_seg = np.ones((ae_bs, 200*64,2))*(-1.)
        ae_ymask_class = np.ones((ae_bs,2))*(-1.)
    class_ymask_seg  = np.ones((seg_class_bs,200*64,2))*(-1.)
    seg_ymask_class = np.ones((seg_class_bs,2))*(-1.)
    while 1:
        seg_rand_ind = np.random.choice(range(segx.shape[0]), size = seg_class_bs)
        class_rand_ind = np.random.choice(range(classx.shape[0]), size = seg_class_bs)
        seg_data_x =  rand_unif_scale(rand_flip_seg(segx[seg_rand_ind]))
        class_data_x = rand_unif_scale(rand_flip_input(classx[class_rand_ind]))
        seg_data_y = segy[seg_rand_ind]
        class_data_y = classy[class_rand_ind]
        if train_ae:
            ae_rand_ind = np.random.choice(range(aex.shape[0]), size = ae_bs)
            ae_data = rand_unif_scale(rand_flip_input(aex[ae_rand_ind]))
            all_input = np.concatenate((seg_data_x, class_data_x, ae_data),0)
            seg_output = np.concatenate(( seg_data_y, class_ymask_seg, ae_ymask_seg),0)
            class_output  = np.concatenate(( seg_ymask_class, class_data_y, ae_ymask_class),0)
            yield all_input, [class_output, seg_output, all_input ]
        else:
            all_input = np.concatenate((seg_data_x, class_data_x),0)
            seg_output = np.concatenate(( seg_data_y, class_ymask_seg),0)
            class_output  = np.concatenate(( seg_ymask_class, class_data_y),0)
            yield all_input, [class_output, seg_output]

def val_gen_masked(aex, segx, segy, classx, classy, batch_size = 32, train_ae = True):
    if train_ae:
        ae_ymask_seg = np.ones((batch_size*2, 200*64,2))*(-1)
        ae_ymask_class = np.ones((batch_size*2,2))*(-1)
        ae_ind_all = np.array(range((aex.shape[0]))*2)
    class_ymask_seg  = np.ones((batch_size,200*64,2))*(-1)
    seg_ymask_class = np.ones((batch_size,2))*(-1)
    class_ind_all = np.arange(classx.shape[0])
    seg_ind_all = np.array(range(segx.shape[0])*11)
    while 1:
        for i in range(80):
            seg_ind = np.arange(i*32,(i+1)*32)
            class_ind = np.arange(i*32,(i+1)*32)
            seg_data_x =  segx[seg_ind_all[seg_ind]]
            class_data_x = classx[class_ind_all[class_ind]]
            seg_data_y = segy[seg_ind_all[seg_ind]]
            class_data_y = classy[class_ind_all[class_ind]]
            if train_ae:
                ae_ind = np.arange(i*64,(i+1)*64)
                ae_data = aex[ae_ind_all[ae_ind]]
                all_input = np.concatenate(( seg_data_x, class_data_x, ae_data),0)
                seg_output = np.concatenate(( seg_data_y, class_ymask_seg, ae_ymask_seg),0)
                class_output  = np.concatenate(( seg_ymask_class, class_data_y, ae_ymask_class),0)
                yield all_input, [class_output, seg_output, all_input]
            else:
                all_input = np.concatenate((seg_data_x, class_data_x),0)
                seg_output = np.concatenate((seg_data_y, class_ymask_seg),0)
                class_output  = np.concatenate((seg_ymask_class, class_data_y),0)
                yield all_input, [class_output, seg_output]




all_gen = gen_masked(X_train, X_train_seg, Y_train_seg, X_train_class, Y_train_class, train_ae = train_ae)
val_gen = val_gen_masked(X_val, X_val_seg, Y_val_seg, X_val_class, Y_val_class, train_ae = train_ae)
all_in, out_list = all_gen.next()
val_in, val_list = val_gen.next()
print("Experiment: {}".format(experiment))
print("Seed: {}".format(seed))


encoder = ss_models.encoder_fc(h, img_size_row, img_size_col, channels, n = n)
classifier = ss_models.classifier_dilation2(h, img_size_row, img_size_col, channels, n = n, model_complexity = .25)
segmenter = ss_models.segmenter_fc(h, img_size_row, img_size_col, channels, n = n)
decoder = ss_models.decoder_fc(h, img_size_row, img_size_col, channels, n = n)

if train_ae:
    ss_model = ss_models.ss_mask(encoder, decoder, classifier, segmenter)
    ss_model.compile( loss={"class_out":build_masked_loss(weighted_class_crossentropy), "seg_out":build_masked_loss(weighted_pixelwise_crossentropy), "ae_out":mae_weighted} ,
        optimizer=nadam_opt_seg_class, metrics = {"class_out":class_acc})#, "seg_out":masked_accuracy, "ae_out":"mse"})
    ss_model.fit_generator(all_gen,
                steps_per_epoch=batches_per_epoch,
                epochs=100,
                shuffle=True,
                validation_data = val_gen,
                validation_steps = 80,
                callbacks=[estp, mchp, rlrp],
                verbose=2)
else:
    ss_model = ss_models.s_mask(encoder, classifier, segmenter)
    ss_model.compile( loss = {"class_out":build_masked_loss(weighted_class_crossentropy), "seg_out":build_masked_loss(weighted_pixelwise_crossentropy)},
        optimizer=nadam_opt_seg_class, metrics = {"class_out":class_acc})#, metrics = {"class_out":class_acc_mask_subset, "seg_out":seg_acc_mask_subset})
    ss_model.fit_generator(all_gen,
                steps_per_epoch=batches_per_epoch,
                epochs=100,
                shuffle=True,
                validation_data = val_gen,
                validation_steps = 80,
                callbacks=[estp, mchp, rlrp],
                verbose=2)
