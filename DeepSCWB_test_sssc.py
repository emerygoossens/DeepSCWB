import sys
sys.path.append("/code/")
from code.data_helpers import *
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import scipy.stats as stats    

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

data_points = 10000
X_test = np.load("data/X_test_sc.npy")[:data_points]
Y_test_seg = np.load("data/Y_seg_test_sc.npy")[:data_points]
Y_test_class = np.load("data/Y_class_test_sc.npy")[:data_points]

sys_arg = str(sys.argv[1])
if sys_arg == "all":
    model_id = "FC37FE"
    experiment = "All Labels"
elif sys_arg == "some":
    model_id = "9A9916"
    experiment = "Some Labels"
elif sys_arg == "few":
    model_id = "3343DC"
    experiment = "Few Labels"



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


model = load_model(model_name, custom_objects={'weighted_pixelwise_crossentropy': weighted_pixelwise_crossentropy, "weighted_class_crossentropy":weighted_class_crossentropy})
preds = model.predict(X_test[:data_points])


y_pred_seg_test = preds[0][:,:,0]
y_pred_seg_reshape = np.reshape(y_pred_seg_test,(y_pred_seg_test.shape[0],200,64))

good_test_ind = np.where(Y_test_class[:,0]==0)[0]
bad_test_ind = np.where(Y_test_class[:,0]==1)[0]

num_cols = min(len(good_test_ind),100)

bad_test_img = X_test[bad_test_ind,:,:,0]
bad_pred_seg = y_pred_seg_reshape[bad_test_ind]*255
bad_img_out = np.hstack((bad_test_img[0],bad_pred_seg[0]))
for c in range(1,num_cols):
    bad_img_out = np.hstack((bad_img_out, bad_test_img[c],bad_pred_seg[c]))

good_img_mean = X_test[good_test_ind,:,:,0].mean((1,2))
good_test_img = np.subtract(X_test[good_test_ind,:,:,0] , good_img_mean[:, None, None])**1.5 -50
good_pred_seg = y_pred_seg_reshape[good_test_ind]*255
good_img_out = np.hstack((good_test_img[0],good_pred_seg[0]))
for c in range(1,num_cols):
    good_img_out = np.hstack((good_img_out, good_test_img[c],good_pred_seg[c]))

good_bad_img_out = np.vstack((good_img_out, bad_img_out))
np.ma.masked_where(good_pred_seg[0] >.5, good_test_img[0] )

cv2.imwrite("images/seg_good_bad_img_pred_test{}.png".format(model_id), good_bad_img_out)






# cv2.imwrite("images/seg_good_img_pred_test.png", y_pred_seg_reshape[good_test_ind[0]]*255)


# class_pred = np.argmax(preds, axis = 1)
class_pred = np.zeros(preds[1].shape[0])
class_pred[np.where(preds[1][:,0]<.1)[0]] = 1
# class_pred = np.argmax(preds[1], axis = 1)
class_true = np.argmax(Y_test_class, axis = 1)

same_pred_ind = np.where(class_pred == class_true)[0];len(same_pred_ind)
diff_pred_ind = np.where(class_pred != class_true)[0]; len(diff_pred_ind)
pred_good_true_bad = np.intersect1d(np.where(class_pred == 1)[0], diff_pred_ind) ; len(pred_good_true_bad)
pred_bad_true_good = np.intersect1d(np.where(class_pred == 0)[0], diff_pred_ind) ; len(pred_bad_true_good)

accuracy = same_pred_ind.shape[0]/float(len(class_pred)); accuracy

X_pred_good_true_bad = X_test[pred_good_true_bad,:]

pred_good_true_bad_img = X_pred_good_true_bad[0,:]
pred_good_true_bad_img = np.squeeze(pred_good_true_bad_img)
for j in range(1, X_pred_good_true_bad.shape[0]):
    pred_good_true_bad_img = np.hstack((pred_good_true_bad_img,np.squeeze(X_pred_good_true_bad[j,:])))

cv2.imwrite("images/X_pred_good_true_bad_{}.png".format(model_id), pred_good_true_bad_img)


X_pred_bad_true_good = X_test[pred_bad_true_good,:]

pred_bad_true_good_img = X_pred_bad_true_good[0,:]
pred_bad_true_good_img = np.squeeze(pred_bad_true_good_img)
for j in range(1, X_pred_bad_true_good.shape[0]):
    pred_bad_true_good_img = np.hstack((pred_bad_true_good_img,np.squeeze(X_pred_bad_true_good[j,:])))

cv2.imwrite("images/X_pred_bad_true_good_{}.png".format(model_id), pred_bad_true_good_img)



seg_pred = preds[0]

good_seg_pred = seg_pred[np.where(Y_test_class[:,0] == 0)[0],:,0]
good_seg_pred = np.reshape(good_seg_pred, (good_seg_pred.shape[0],200,64))*256
good_X_test = X_test[np.where(Y_test_class[:,0] == 0)[0],:,:,0]

seg_good_img = np.hstack((good_X_test[0,],good_seg_pred[0,:,:]))
for j in range(1, 200):
    seg_good_img = np.hstack((seg_good_img,np.squeeze(good_X_test[j,:])))


cv2.imwrite("images/X_good_seg_pred_{}.png".format(model_id), seg_good_img)



# X_train = np.load("data/X_train_sc.npy")
# Y_seg_train = np.load("data/Y_seg_train_sc.npy")
# Y_class_train = np.load("data/Y_class_train_sc.npy")

# good_seg_pred = Y_seg_train[np.where(Y_class_train[:,1] == 1)[0],:,0]
# good_seg_pred = np.reshape(good_seg_pred, (good_seg_pred.shape[0],200,64))*256
# good_X_train = X_train[np.where(Y_class_train[:,1] == 1)[0],:,:,0]

# seg_good_img = np.hstack((good_X_train[0,],good_seg_pred[0,:,:]))
# for j in range(1, 200):
#   seg_good_img = np.hstack((good_seg_pred[j,],np.squeeze(good_X_train[j,:])))

# cv2.imwrite("images/X_good_seg_label.png", seg_good_img)

output_mat = np.zeros((good_seg_pred.shape[0]))
for j in range(good_seg_pred.shape[0]):
    an_image = np.copy(good_X_test[j,:,7:57])
    border_mean_rows = np.copy(an_image[:,(range(5), range(45,50))].mean((1,2)))
    background_sub_image = np.subtract(an_image, border_mean_rows[:, None])
    background_sub_image[background_sub_image <0] = 0
    good_pixels = np.where(good_seg_pred[j,:,7:57:]>.5)
    # background_pixels = np.where(good_seg_pred[j,:,:]<.5)
    protein_sum = background_sub_image[good_pixels].sum()
    output_mat[j] = protein_sum



output_mat = output_mat*(2*655.35/255.0)


# good_seg_pred = y_pred_seg_reshape[good_test_ind][:,:,7:57]


# output_mat = np.zeros((good_seg_pred.shape[0]))
# for j in range(good_seg_pred.shape[0]):
#     an_image = np.copy(good_test_img[j,:,7:57])
#     border_mean_rows = np.copy(an_image[:,(range(5), range(45,50))].mean((1,2)))
#     background_sub_image = np.subtract(an_image, border_mean_rows[:, None])
#     background_sub_image[background_sub_image <0] = 0
#     good_pixels = np.where(good_seg_pred[j,:,:]>.5)
#     # background_pixels = np.where(good_seg_pred[j,:,:]<.5)
#     protein_sum = background_sub_image[good_pixels].sum()
#     output_mat[j] = protein_sum

    # an_image[background_pixels] = 0
    # protein_sum_rows = an_image.sum(-1)
    # background_pixels_border = background_pixels[]
    # border_ind_left = np.where(background_pixels[1] <5)
    # border_ind_right = np.where(background_pixels[1] >194)

    # background_median = np.median(an_image[background_pixels])
    # background_mean = np.mean(an_image[background_pixels])
    # protein_values = (an_image[good_pixels] - background_median)
    # protein_values[protein_values <0] = 0
    # protein_sum = protein_values.sum()
    # output_mat[j,0] = protein_sum
    # output_mat[j,1] = background_median
    # output_mat[j,2] = background_mean
    # output_mat[j,3] = an_image[good_pixels].mean()


# fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data)


# np.save("data/protein_expression_{}.npy".format(model_id), output_mat)


scwb_measurements = output_mat[:]
n, bins, patches = plt.hist(scwb_measurements, 100, normed=1, facecolor='green', alpha=0.75)
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(scwb_measurements)
y1 = stats.gamma.pdf(bins, a=fit_alpha, scale=fit_beta)

mu = round(scwb_measurements.mean(),1)
sigma = round(scwb_measurements.std(),1)
median = round(np.median(scwb_measurements),1)
# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.plot(bins, y1, "r--", label=(r'$\alpha=0.88, \beta=12374$')) 


# plt.xlabel('Protein Expression')
# plt.ylabel('Probability')
# plt.title('{}: Mean = {}, Std Err = {}, Median = {}'.format(experiment,mu,sigma, median), y=1.07)
# plt.axis([0, 80000, 0, 0.0002])
# plt.grid(True)
# plt.savefig("images/scwb_measurements_{}_{}.png".format(model_id, experiment))
# plt.close()



# np.save("../data/scwb_measurements", scwb_measurements)

# nms_boxes[j,:]
# nms_lower_q[j,:]
# nms_upper_q[j,:]

def weighted_pixelwise_crossentropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = np.sum(np.multiply(y_true * np.log(y_pred), np.array([good_weight_seg,1],"float32")),-1)
    return - np.mean(class_weight_loss)

def weighted_class_crossentropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 10e-8, 1. - 10e-8)
    class_weight_loss = np.sum(np.multiply(y_true * np.log(y_pred), np.array([1*good_weight_seg_class_ratio,good_weight_class*good_weight_seg_class_ratio],"float32")),-1)
    return - np.mean(class_weight_loss)

# np.save("../data/overlap_boxes.npy", overlap_boxes)
y_seg_pred = preds[0]
y_seg_pred_class = np.argmax(y_seg_pred, -1)
seg_accuracy = len(np.where(y_seg_pred_class == np.argmax(Y_test_seg,-1))[0])/(10000.*200*64)

pixel_loss = weighted_pixelwise_crossentropy_np(Y_test_seg, y_seg_pred)
class_loss = weighted_class_crossentropy_np(Y_test_class, preds[1])

results_table = np.zeros((7,1))

results_table[0]  = accuracy
results_table[1] = seg_accuracy
results_table[2] = class_loss
results_table[3] = pixel_loss
results_table[4] = mu
results_table[5] = sigma
results_table[6] = median
np.save("data/results_table_{}_{}.npy".format(model_id, experiment), results_table)

print(results_table)


good_test_img = np.subtract(X_test[good_test_ind,:,:,0] , good_img_mean[:, None, None])**1.5 -50
good_pred_seg = y_pred_seg_reshape[good_test_ind]*255



fig, ax = plt.subplots()

# gray = ax.imshow(np.hstack(good_test_img[0,:,7:57]), cmap='gray_r', alpha =1)
seg = ax.imshow(np.transpose(good_seg_pred[0,:,:]), cmap='summer_r', alpha = 1)

# cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
cbar = fig.colorbar(seg, cax = cbaxes, orientation = "horizontal",fraction=0.07,anchor=(1.0,0.0))
# cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
cbar.ax.tick_params(labelsize=18)

plt.axis('off')


plt.savefig("images/seg_map_single_{}.png".format(model_id))


good_recon_pred = preds[2][np.where(Y_test_class[:,0] == 0)[0],:,7:57,0]


fig, ax = plt.subplots()

# gray = ax.imshow(np.hstack(good_test_img[0,:,7:57]), cmap='gray_r', alpha =1)
recon = ax.imshow(good_recon_pred[0,:,:], cmap='gray_r', alpha = 1, vmin=good_recon_pred[0,:,:].min(), vmax=good_recon_pred[0,:,:].max())

# cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
# cbar = fig.colorbar(seg, cax = cbaxes, orientation = "horizontal",anchor=(1.0,0.0))
# cbar.ax.set_yticklabels(['0', '1'])  # vertically oriented colorbar
# cbar.ax.tick_params(labelsize=18)

plt.axis('off')


plt.savefig("images/recon_single_{}.png".format(model_id))





fig, ax = plt.subplots()

# gray = ax.imshow(np.hstack(good_test_img[0,:,7:57]), cmap='gray_r', alpha =1)
input_image = ax.imshow(good_X_test[0,:,7:57], cmap='gray_r', alpha = 1, vmin=good_recon_pred[0,:,:].min(), vmax=good_recon_pred[0,:,:].max())


plt.axis('off')
plt.savefig("images/input_sing_{}.png".format(model_id))


preds[1][good_test_ind[1]]

# plt.figure(figsize = (200, 64))
# # plt.subplot(1,3,1)
# plt.imshow( np.asarray(good_test_img[0]) )
# plt.subplot(1,3,2)
# plt.imshow( imclass )
# plt.subplot(1,3,3)
# plt.imshow( np.asarray(crpim) )
# masked_imclass = np.ma.masked_where(imclass == 0, imclass)
# #plt.imshow( imclass, alpha=0.5 )
# plt.imshow( masked_imclass, alpha=0.5 )

# plt.figure(figsize = (200, 64))

# rgb = np.zeros((200, 64, 2), dtype=np.uint8)

# plt.plot(good_test_img[0])
# plt.savefig("images/an_image_{}_{}.png".format(model_id, experiment))
# plt.close()
