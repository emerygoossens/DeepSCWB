import sys
sys.path.append("/code/")
from code.data_helpers import *
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
start = time.time()


make_data = True
if make_data:
    images, labels, filenames, indices  = load_data("EG_CSHL/scwb_images","EG_CSHL/emery_summary.csv")
    images = np.reshape(images, (images.shape[0],200,50))
    images = (images + 32768.0)*(255.0/65535)
    padded_images = np.zeros((images.shape[0],200,64,1))
    padded_images[:,:,7:57,0] = images
    names_indices = np.hstack((filenames,indices))
    sorted_names = names_indices[np.argsort(names_indices[:,0]),:]
    sorted_indices = sorted_names[:,1].astype("int32")
    np.save("data/X_test_EG_CSHL.npy",padded_images[sorted_indices])
    np.save("data/Y_test_EG_CSHL.npy",labels)
    X_test_EG_CSHL = np.load("data/X_test_EG_CSHL.npy")
    Y_test_EG_CSHL = np.load("data/Y_test_EG_CSHL.npy")
else:
    X_test_EG_CSHL = np.load("data/X_test_EG_CSHL.npy")
    Y_test_EG_CSHL = np.load("data/Y_test_EG_CSHL.npy")



# X_test_EG_CSHL = X_test_EG_CSHL[sorted_indices]
# Y_test_EG_CSHL = Y_test_EG_CSHL[sorted_indices]

model_id = "FC37FE"
model_name = "saved_models/model_{}/model_{}".format(model_id, model_id)


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

# model = load_model(model_name)
# preds = model.predict(X_test_goldstd[:,:,7:57,:])

auc_data = np.loadtxt("EG_CSHL/emery_summary.csv", dtype = 'str', delimiter = ",", skiprows = 1)[:,2].astype("float32")
label_data = np.loadtxt("EG_CSHL/emery_summary.csv", dtype = 'str', delimiter = ",", skiprows = 1)
labels = label_data[:,1].astype("int32")
model = load_model(model_name, custom_objects={'weighted_pixelwise_crossentropy': weighted_pixelwise_crossentropy, "weighted_class_crossentropy":weighted_class_crossentropy})
preds = model.predict(X_test_EG_CSHL)


end = time.time()
print(end - start)
X_data = X_test_EG_CSHL
# def quantify_scwb(model, X_data, labels):

preds = model.predict(X_data)
good_pred_test_ind = np.where(preds[1][:,0]<.1)[0]
good_true_test_ind = np.where(labels >.5)[0]

y_pred_seg_test = preds[0][:,:,0]
y_pred_seg_reshape = np.reshape(y_pred_seg_test,(y_pred_seg_test.shape[0],200,64))

good_ind_uni  = np.union1d(good_pred_test_ind,good_true_test_ind )
good_ind_int = np.intersect1d(good_pred_test_ind,good_true_test_ind )
good_ind_DL_only = np.setdiff1d(good_pred_test_ind, good_true_test_ind)
good_ind_AUC_only = np.setdiff1d( good_true_test_ind, good_pred_test_ind)

good_test_img = X_data[good_ind_uni,:,:,0]
good_seg_pred = y_pred_seg_reshape[good_ind_uni][:,:,7:57]
good_true_labels_uni = labels[good_ind_uni]
good_pred_labels_uni = preds[1][good_ind_uni]
auc_good = auc_data[good_ind_uni]

good_pred_ind_uni = np.where(good_pred_labels_uni[:,0] <.1)[0]
good_true_ind_uni = np.where(good_true_labels_uni >.5)[0]


# good_ind_uni_plot  = np.union1d(good_pred_ind_uni,good_true_ind_uni )
# good_ind_int_plot = np.intersect1d(good_pred_ind_uni,good_true_ind_uni )
# good_ind_DL_only_plot = np.setdiff1d(good_pred_ind_uni, good_true_ind_uni)
# good_ind_AUC_only_plot = np.setdiff1d( good_true_ind_uni, good_pred_ind_uni)

output_mat = np.zeros((good_seg_pred.shape[0],1))
for j in range(good_seg_pred.shape[0]):
    an_image = np.copy(good_test_img[j,:,7:57])
    border_mean_rows = np.copy(an_image[:,(range(5), range(45,50))].mean((1,2)))
    background_sub_image = np.subtract(an_image, border_mean_rows[:, None])
    background_sub_image[background_sub_image <0] = 0
    good_pixels = np.where(good_seg_pred[j,:,:]>.5)
    # background_pixels = np.where(good_seg_pred[j,:,:]<.5)
    protein_sum = background_sub_image[good_pixels].sum()
    output_mat[j,0] = protein_sum



output_mat = output_mat*(2*655.35/255.0)

line = plt.figure(figsize = (9,6))


plot_int_ind = np.intersect1d(good_pred_ind_uni, good_true_ind_uni)
plot_dl_good_ind = np.setdiff1d(good_pred_ind_uni, good_true_ind_uni)
plot_auc_good_ind = np.setdiff1d(good_true_ind_uni,good_pred_ind_uni)


both_good, = plt.plot(output_mat[plot_int_ind],auc_good[plot_int_ind], "o", color = "b", label = "Both Selected")
dl_good_only, = plt.plot(output_mat[plot_dl_good_ind],auc_good[plot_dl_good_ind], "o",color ="r", label= "DL Selected Only")
man_good_only, = plt.plot(output_mat[plot_auc_good_ind],auc_good[plot_auc_good_ind], "o",color = "g", label = "Manually Selected Only")


plt.xlabel('Deep Learning Segmentation', fontsize = 24)
plt.ylabel('Manual Gaussian Fitting', fontsize = 24, x = .9)
line.subplots_adjust(top=.9)

plt.title('Intensity Comparison of Protein Targets (AFU)', y=1.02, fontsize = 26)
plt.axis([-10000, 200000, -10000, 200000], fontsize = 22)
plt.legend(numpoints = 1, loc = 2, fontsize = 20)
plt.tight_layout()

plt.savefig("images/quant_auc_scatter_int_good_{}.png".format(model_id))
plt.close()




# good_test_img = X_data[good_pred_test_ind,:,:,0]
# good_seg_pred = y_pred_seg_reshape[good_pred_test_ind][:,:,7:57]


# output_mat = np.zeros((good_seg_pred.shape[0],1))
# for j in range(good_seg_pred.shape[0]):
#     an_image = np.copy(good_test_img[j,:,7:57])
#     border_mean_rows = np.copy(an_image[:,(range(5), range(45,50))].mean((1,2)))
#     background_sub_image = np.subtract(an_image, border_mean_rows[:, None])
#     background_sub_image[background_sub_image <0] = 0
#     good_pixels = np.where(good_seg_pred[j,:,:]>.5)
#     # background_pixels = np.where(good_seg_pred[j,:,:]<.5)
#     protein_sum = background_sub_image[good_pixels].sum()
#     output_mat[j,0] = protein_sum








fig, ax = plt.subplots(figsize = (9,6))

gray = ax.imshow(np.hstack(good_test_img[0:7,:,7:57]), cmap='gray_r', alpha =1)
seg = ax.imshow(np.hstack(good_seg_pred[0:7,:,:]), cmap='summer_r', alpha = .25)

cbar = fig.colorbar(seg)
# cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
cbar.ax.tick_params(labelsize=18)

plt.axis('off')
plt.title('Protein Segmentation Probabilities', y=1.02, fontsize = 26)


plt.savefig("images/seg_map_{}.png".format(model_id))

# fig, ax = plt.subplots()

# plt.figure()
# ax = plt.gca()

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np

# plt.figure()
# ax = plt.gca()
# im = ax.imshow(np.arange(100).reshape((10,10)))

# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im, cax=cax)


# data = np.clip(randn(250, 250), -1, 1)

# cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
# ax.set_title('Gaussian noise with vertical colorbar')



# plt.xlabel('Deep Learning Segmentation')
# plt.ylabel('AUC')
# plt.title('Comparison of Quantification Algorithms: DL Predicted Good', y=1.07)


# plt.plot(output_mat,auc_data[good_pred_test_ind], "o",)
# plt.close()



