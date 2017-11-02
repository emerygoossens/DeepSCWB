import sys
sys.path.append("/code/")
from code.data_helpers import *
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2



make_data = False
if make_data:
	images, labels = load_data("goldstd","goldstd/goldstd_20170616.csv")
	images = np.reshape(images, (images.shape[0],200,50))
	images = (images + 32768.0)*(256.0/65535)
	padded_images = np.zeros((images.shape[0],200,64,1))
	padded_images[:,:,7:57,0] = images
	np.save("data/X_test_goldstd.npy",padded_images)
	np.save("data/Y_test_goldstd.npy",labels)
	X_test_goldstd = np.load("data/X_test_goldstd.npy")
	Y_test_goldstd = np.load("data/Y_test_goldstd.npy")
else:
	X_test_goldstd = np.load("data/X_test_goldstd.npy")
	Y_test_goldstd = np.load("data/Y_test_goldstd.npy")


model_id = "A7C2A5"
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
preds = model.predict(X_test_goldstd)


y_pred_seg_test = preds[0][:,:,0]
y_pred_seg_reshape = np.reshape(y_pred_seg_test,(y_pred_seg_test.shape[0],200,64))

good_test_ind = np.where(Y_test_goldstd[:,0]==0)[0]


num_cols = 100

good_test_img = X_test_goldstd[good_test_ind,:,:,0]
good_pred_seg = y_pred_seg_reshape[good_test_ind]*255
starter = np.hstack((good_test_img[0],good_pred_seg[0]))
for c in range(1,num_cols):
    starter = np.hstack((starter, good_test_img[c],good_pred_seg[c]))

cv2.imwrite("images/seg_good_img_pred2.png", starter)

# cv2.imwrite("images/seg_good_img_pred_test.png", y_pred_seg_reshape[good_test_ind[0]]*255)


# class_pred = np.argmax(preds, axis = 1)
class_pred = np.argmax(preds[1], axis = 1)
class_true = np.argmax(Y_test_class, axis = 1)

same_pred_ind = np.where(class_pred == class_true)[0];len(same_pred_ind)
diff_pred_ind = np.where(class_pred != class_true)[0]; len(diff_pred_ind)
pred_good_true_bad = np.intersect1d(np.where(class_pred == 1)[0], diff_pred_ind); len(pred_good_true_bad)
pred_bad_true_good = np.intersect1d(np.where(class_pred == 0)[0], diff_pred_ind); len(pred_bad_true_good)

accuracy = same_pred_ind.shape[0]/float(len(class_pred));accuracy

X_pred_good_true_bad = X_test_goldstd[pred_good_true_bad,:]

pred_good_true_bad_img = X_pred_good_true_bad[0,:]
pred_good_true_bad_img = np.squeeze(pred_good_true_bad_img)
for j in range(1, X_pred_good_true_bad.shape[0]):
	pred_good_true_bad_img = np.hstack((pred_good_true_bad_img,np.squeeze(X_pred_good_true_bad[j,:])))

cv2.imwrite("images/X_pred_good_true_bad.png", pred_good_true_bad_img)


X_pred_bad_true_good = X_test_goldstd[pred_bad_true_good,:]

pred_bad_true_good_img = X_pred_bad_true_good[0,:]
pred_bad_true_good_img = np.squeeze(pred_bad_true_good_img)
for j in range(1, X_pred_bad_true_good.shape[0]):
	pred_bad_true_good_img = np.hstack((pred_bad_true_good_img,np.squeeze(X_pred_bad_true_good[j,:])))

cv2.imwrite("images/X_pred_bad_true_good.png", pred_bad_true_good_img)



seg_pred = preds[0]

good_seg_pred = seg_pred[np.where(Y_test_goldstd[:,0] == 0)[0],:,0]
good_seg_pred = np.reshape(good_seg_pred, (good_seg_pred.shape[0],200,64))*256
good_X_test = X_test_goldstd[np.where(Y_test_goldstd[:,0] == 0)[0],:,:,0]

seg_good_img = np.hstack((good_X_test[0,],good_seg_pred[0,:,:]))
for j in range(1, 200):
	seg_good_img = np.hstack((seg_good_img,np.squeeze(good_X_test[j,:])))

cv2.imwrite("images/X_good_seg_pred.png", good_seg_pred[0])



X_train = np.load("data/X_train_sc.npy")
Y_seg_train = np.load("data/Y_seg_train_sc.npy")
Y_class_train = np.load("data/Y_class_train_sc.npy")

good_seg_pred = Y_seg_train[np.where(Y_class_train[:,1] == 1)[0],:,0]
good_seg_pred = np.reshape(good_seg_pred, (good_seg_pred.shape[0],200,64))*256
good_X_train = X_train[np.where(Y_class_train[:,1] == 1)[0],:,:,0]

seg_good_img = np.hstack((good_X_train[0,],good_seg_pred[0,:,:]))
for j in range(1, 200):
	seg_good_img = np.hstack((good_seg_pred[j,],np.squeeze(good_X_train[j,:])))

cv2.imwrite("images/X_good_seg_label.png", seg_good_img)

