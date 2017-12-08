from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import numpy as np
import os
from scipy import signal
import random
import tensorflow as tf


def model_id_ss(experiment, seed,es_patience = 50, rl_patience = 10, rl_factor = .9, monitor_es = "val_loss", path = "./"):
    model_name = ''.join(random.choice('0123456789ABCDEF') for i in range(6))
    ensure_dir("{}saved_models/".format(path))
    model_path = "{}saved_models/{}_{}/".format(path,experiment, seed)
    ensure_dir(model_path)
    model_path = model_path + model_name + "/"
    ensure_dir(model_path)
    estp = EarlyStopping(monitor=monitor_es,patience=es_patience,verbose=2)
    mchp = ModelCheckpoint("{}/model_{}".format(model_path, model_name), save_best_only = True, save_weights_only=True)
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
    w=np.ones([window_len,window_len/2],'d')
    y = signal.convolve2d(x, w/w.sum(), boundary='fill', mode='same')
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



def add_high_intensity_noise(data, noise_max_scale = 1.5, noise_per_image =2, noise_width =3, x_start = 7, x_end = 57, y_start = 0, y_end = 200):
    data_copy = np.copy(data)
    for j in range(data_copy.shape[0]):
        for k in range(noise_per_image):
            sample_x = np.random.choice(range(x_start+noise_width,x_end - noise_width), size = 1)[0]
            sample_y = np.random.choice(range(noise_width,200 -noise_width), size = 1)[0]
            data_copy[j,sample_y - noise_width:sample_y + noise_width, sample_x - noise_width:sample_x + noise_width,0 ] = np.clip(data_copy[j].max()*noise_max_scale*np.ones((noise_width*2, noise_width*2)),0,256)
    return data_copy


def add_low_intensity_noise(data, noise_max_scale = 1.5, noise_per_image =2, noise_width =3, x_start = 7, x_end = 57, y_start = 0, y_end = 200):
    data_copy = np.copy(data)
    for j in range(data_copy.shape[0]):
        for k in range(noise_per_image):
            sample_x = np.random.choice(range(x_start+noise_width,x_end - noise_width), size = 1)[0]
            sample_y = np.random.choice(range(noise_width,200 -noise_width), size = 1)[0]
            data_copy[j,sample_y - noise_width:sample_y + noise_width, sample_x - noise_width:sample_x + noise_width,0 ] = np.clip((data_copy[j,:,x_start:x_end,:].min()/noise_max_scale)*np.ones((noise_width*2, noise_width*2)),0,256)
    return data_copy

# def iou(y_true, y_pred):
#     # true positive / (true positive + false positive + false negative)
    


def quantify_scwb(input_data, model):
    preds = model.predict(input_data)
    good_pred_test_ind = np.where(preds[0][:,0]<.5)[0]
    y_pred_seg_test = preds[1][:,:,0]
    y_pred_seg_reshape = np.reshape(y_pred_seg_test,(y_pred_seg_test.shape[0],200,64))
    good_test_img = input_data[good_pred_test_ind,:,:,0]
    good_seg_pred = y_pred_seg_reshape[good_pred_test_ind][:,:,7:57]
    output_mat = np.zeros((good_seg_pred.shape[0],1))
    for j in range(good_seg_pred.shape[0]):
        an_image = np.copy(good_test_img[j,:,7:57])
        border_mean_rows = np.copy(an_image[:,(range(5), range(45,50))].mean((1,2)))
        background_sub_image = np.subtract(an_image, border_mean_rows[:, None])
        background_sub_image[background_sub_image <0] = 0
        good_pixels = np.where(good_seg_pred[j,:,:]>.5)
        protein_sum = background_sub_image[good_pixels].sum()
        output_mat[j,0] = protein_sum

    output_mat = output_mat*(2*655.35/255.0)
    return output_mat   


def load_sssc_model(model_name):
    good_weight_seg = 4.5
    good_weight_class = 9.7
    good_weight_seg_class_ratio =.47
    def weighted_pixelwise_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
        class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([good_weight_seg,1],"float32")),-1)
        return - tf.reduce_mean(class_weight_loss)
    def weighted_class_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
        class_weight_loss = tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), tf.constant([1*good_weight_seg_class_ratio,good_weight_class*good_weight_seg_class_ratio],"float32")),-1)
        return - tf.reduce_mean(class_weight_loss)
    model = load_model(model_name, custom_objects={'weighted_pixelwise_crossentropy': weighted_pixelwise_crossentropy, "weighted_class_crossentropy":weighted_class_crossentropy})
    return model 

