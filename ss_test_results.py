import sys
sys.path.append("/code/")
# from code.data_helpers import *
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import scipy.stats as stats    
import ss_models
from code.ae_helpers import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

X_test = np.load("data/X_test_ss.npy")
Y_test_class = np.load("data/Y_test_class_ss.npy")
Y_test_seg = np.load("data/Y_test_seg_ss.npy")
X_test_seg = np.load("data/X_test_seg_ss.npy")

Y_val_class = np.load("data/Y_val_class_ss.npy")
Y_val_seg = np.load("data/Y_val_seg_ss.npy")

X_val_seg = np.load("data/X_val_seg_ss.npy")
X_val_class = np.load("data/X_val_class_ss.npy")


n = 32
h = n*4 #Autoencoder hidden representation

img_size_row = 200 
img_size_col = 64
channels = 1 


experiment = str(sys.argv[1])

print("Testing experiement: {}".format(experiment))


if experiment == "seg_class_all":
    data_percent = 1
    train_ae = False
    train_seg = True
    train_class = True
elif experiment == "seg_class_ae_all":
    data_percent = 1
    train_ae = True
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





# model_names[0] = "1E68F3"
good_weight_seg = 4.5
good_weight_class = 9.7
good_weight_seg_class_ratio = 1


def precision_recall_iou_class(y_true, y_pred):
    true_pos_inds = np.where(np.argmax(y_true,-1) == 1)[0]
    true_neg_inds = np.where(np.argmax(y_true,-1) == 0)[0]
    pred_pos_inds = np.where(np.argmax(y_pred,-1) == 1)[0]
    pred_neg_inds = np.where(np.argmax(y_pred,-1) == 0)[0]
    corr_pos_inds = np.intersect1d(true_pos_inds, pred_pos_inds)
    fals_pos_inds = np.setdiff1d(pred_pos_inds,true_pos_inds)
    fals_neg_inds = np.intersect1d(pred_neg_inds, true_pos_inds)
    precision = float(len(corr_pos_inds))/(len(corr_pos_inds) + len(fals_pos_inds))
    recall = float(len(corr_pos_inds))/(len(true_pos_inds))
    iou = float(len(corr_pos_inds))/(len(corr_pos_inds) + len(fals_pos_inds) + len(fals_neg_inds))
    return precision, recall, iou

def precision_recall_iou_seg(y_true, y_pred):
    true_pos_inds = np.where(np.argmax(y_true,-1) == 0)[0]
    true_neg_inds = np.where(np.argmax(y_true,-1) == 1)[0]
    pred_pos_inds = np.where(np.argmax(y_pred,-1) == 0)[0]
    pred_neg_inds = np.where(np.argmax(y_pred,-1) == 1)[0]
    corr_pos_inds = np.intersect1d(true_pos_inds, pred_pos_inds)
    fals_pos_inds = np.setdiff1d(pred_pos_inds,true_pos_inds)
    fals_neg_inds = np.intersect1d(pred_neg_inds, true_pos_inds)
    precision = float(len(corr_pos_inds))/(len(corr_pos_inds) + len(fals_pos_inds))
    recall = float(len(corr_pos_inds))/(len(true_pos_inds))
    iou = float(len(corr_pos_inds))/(len(corr_pos_inds) + len(fals_pos_inds) + len(fals_neg_inds))
    return precision, recall, iou


def weighted_pixelwise_crossentropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = np.sum(np.multiply(y_true * np.log(y_pred), np.array([good_weight_seg,1],"float32")),-1)
    return - np.mean(class_weight_loss)

def weighted_class_crossentropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = np.sum(np.multiply(y_true * np.log(y_pred), np.array([1*good_weight_seg_class_ratio,good_weight_class*good_weight_seg_class_ratio],"float32")),-1)
    return - np.mean(class_weight_loss)



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


def seg_acc( seg_true, seg_pred):
    pred_seg_outcome = np.argmax(seg_pred, -1)
    return len(np.where(pred_seg_outcome == np.argmax(seg_true,-1))[0])/(seg_true.shape[0]*200.*64)

def class_acc( class_true, class_pred):
    pred_class_outcome = np.argmax(class_pred, -1)
    return len(np.where(pred_class_outcome == np.argmax(class_true,-1))[0])/(float(class_true.shape[0]))




if train_ae:
    encoder = ss_models.encoder_fc(h, img_size_row, img_size_col, channels, n = n)
    classifier = ss_models.classifier_dilation2(h, img_size_row, img_size_col, channels, n = n, model_complexity = .25)
    segmenter = ss_models.segmenter_fc(h, img_size_row, img_size_col, channels, n = n)
    decoder = ss_models.decoder_fc(h, img_size_row, img_size_col, channels, n = n)
    ss_model = ss_models.ss_mask(encoder, decoder, classifier, segmenter)
else:
    encoder = ss_models.encoder_fc(h, img_size_row, img_size_col, channels, n = n)
    classifier = ss_models.classifier_dilation2(h, img_size_row, img_size_col, channels, n = n, model_complexity = .25)
    segmenter = ss_models.segmenter_fc(h, img_size_row, img_size_col, channels, n = n)
    decoder = ss_models.decoder_fc(h, img_size_row, img_size_col, channels, n = n)
    ss_model = ss_models.s_mask(encoder, classifier, segmenter)


num_results = 11

best_model_dict = {}
best_results = np.zeros((5,num_results))

for seed in range(1,6):

    experiment_path = "saved_models/{}_{}/".format(experiment, seed)
    model_names = os.listdir(experiment_path)

    seed_results = np.ones((1,num_results))*10

    for j in range(len(model_names)):
        print(model_names[j])
        try:
            ss_model.load_weights("{}{}/model_{}".format(experiment_path,model_names[j],model_names[j]))
        except IOError:
            continue
        
        class_preds = ss_model.predict(X_test)[0]
        seg_preds = ss_model.predict(X_test_seg)[1]

        class_preds_val = ss_model.predict(X_val_class)[0]
        seg_preds_val = ss_model.predict(X_val_seg)[1]

        class_precision, class_recall, class_iou = precision_recall_iou_class(Y_test_class, class_preds)
        seg_precision, seg_recall, seg_iou = precision_recall_iou_seg(Y_test_seg, seg_preds)


        results_table = np.ones((1,num_results))*10

        class_accuracy = class_acc(Y_test_class, class_preds)
        seg_accuracy = seg_acc(Y_test_seg, seg_preds)

        pixel_loss = weighted_pixelwise_crossentropy_np(Y_test_seg, seg_preds)
        class_loss = weighted_class_crossentropy_np(Y_test_class, class_preds)


        class_accuracy_val = class_acc(class_preds_val, Y_val_class)
        seg_accuracy_val = seg_acc(seg_preds_val, Y_val_seg)
        
        pixel_loss_val = weighted_pixelwise_crossentropy_np(Y_val_seg, seg_preds_val)
        class_loss_val = weighted_class_crossentropy_np(Y_val_class, class_preds_val)

        loss_val = pixel_loss_val + class_loss_val

        results_table[0,0]  = class_accuracy
        results_table[0,1] = seg_accuracy
        results_table[0,2] = class_loss
        results_table[0,3] = pixel_loss
        results_table[0,4] = class_precision
        results_table[0,5] = class_recall
        results_table[0,6] = seg_iou

        scwb_measurements = quantify_scwb(X_test, ss_model)
        mu = round(scwb_measurements.mean(),1)
        sigma = round(scwb_measurements.std(),1)
        median = round(np.median(scwb_measurements),1)
        results_table[0,7] = mu
        results_table[0,8] = sigma
        results_table[0,9] = median

        results_table[0,10] = loss_val

        seed_results = np.concatenate((seed_results, results_table))

        print(results_table)

    seed_results = seed_results[1:,:]

    best_model_ind = np.argmin(seed_results[:,num_results-1])
    best_model_dict[seed] = model_names[best_model_ind]
    best_results[seed -1,:] = seed_results[best_model_ind,:]
    print(best_model_ind)

print(best_results[:,0].mean())

np.save("data/best_results_table_{}.npy".format(experiment), best_results)
np.save("data/best_models_{}.npy".format(experiment), best_model_dict)


#After completion of all experiments
seg_class_all_best = np.load("data/best_results_table_{}.npy".format("seg_class_all"))
seg_class_ae_all_best = np.load("data/best_results_table_{}.npy".format("seg_class_ae_all"))
seg_class_ae_10_best = np.load("data/best_results_table_{}.npy".format("seg_class_ae_10"))
seg_class_ae_1_best = np.load("data/best_results_table_{}.npy".format("seg_class_ae_1"))



