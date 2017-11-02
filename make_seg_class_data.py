
import numpy as np 
import os
from scipy.ndimage.interpolation import rotate
import random
from autoencoder_helpers import *
from scipy import signal
import cv2

from code.ae_helpers import *


data_points = 10000000
pix_to_roll = 15


all_data = reformat(np.load("data/image_dataset_aug.npy"))[:data_points,]
labels = np.load("data/image_labels_aug.npy")[:data_points,]
all_data = (all_data + 32768.0)*(256.0/65535)

X_train = np.zeros((all_data.shape[0], 200,64,1))

X_train[:,:,7:57,:] = all_data

good_ind = np.where(labels[:,1] ==1)[0]
bad_ind = np.where(labels[:,1] ==0)[0]


bad_imgs = X_train[bad_ind,:]


bad_seg_labels = np.zeros((bad_imgs.shape))



# seg_bad_imgs_paddedz = np.ones((bad_imgs.shape[0],1,200,64))/2.
# seg_bad_imgs_paddedo = np.ones((bad_imgs.shape[0],1,200,64))/2.
# seg_bad_imgs_padded = np.hstack((seg_bad_imgs_paddedz,seg_bad_imgs_paddedo))

# seg_bad_imgs_padded = np.reshape(seg_bad_imgs_padded, (seg_bad_imgs_padded.shape[0],2,64*200))
# seg_bad_imgs_padded = np.moveaxis(seg_bad_imgs_padded, -1,1)

seg_bad_imgs = np.zeros((bad_imgs.shape[0],200*64))


good_imgs = X_train[good_ind,:]


seg_good_imgs = np.zeros((good_imgs.shape[0],200,64,1))







for j in range(good_imgs.shape[0]):
    segmented_image = good_imgs[j,:,7:57,0]
    segmented_image = np.clip((segmented_image- np.mean(segmented_image))*50,0, 255)
    segmented_image[:40,:] = 0
    segmented_image[-40:,:] = 0
    segmented_image[:,:20] = 0
    segmented_image[:,-20:] = 0
    segmented_image = smooth(segmented_image)
    segmented_image[np.where(segmented_image >segmented_image.mean())] = 255
    segmented_image[np.where(segmented_image <segmented_image.mean())] = 0
    segmented_image = smooth(segmented_image)
    segmented_image[np.where(segmented_image >segmented_image.mean())] = 255
    segmented_image[np.where(segmented_image <segmented_image.mean())] = 0
    seg_good_imgs[j,:,7:57,0] = segmented_image


# seg_good_imgs_paddedz = np.zeros((seg_good_imgs.shape[0],1,200,64))
# seg_good_imgs_paddedo = np.ones((seg_good_imgs.shape[0],1,200,64))
# seg_good_imgs_padded = np.hstack((seg_good_imgs_paddedz,seg_good_imgs_paddedo))
# # seg_good_imgs = np.reshape(seg_good_imgs,(seg_good_imgs.shape[0],200,64))
# seg_good_imgs_padded[:,0,:,7:57] = seg_good_imgs[:,:,:,0]
# seg_good_imgs_padded[:,1,:,7:57] = np.subtract(np.ones(seg_good_imgs.shape[:3]), seg_good_imgs[:,:,:,0])

img_set1 = good_imgs
img_set2 = seg_good_imgs
imgs = np.hstack((img_set1[0],img_set2[0]))
num_rows = 5
num_cols = 1000
starter = np.hstack((img_set1[0],img_set2[0]))
for c in range(1,num_cols):
    starter = np.hstack((starter, img_set1[c],img_set2[c]))

cv2.imwrite("images/seg_good_img_test.png", starter)

seg_good_imgs = seg_good_imgs/255
seg_good_imgs = np.reshape(seg_good_imgs, (seg_good_imgs.shape[0],200*64))

# for r in range(1,num_rows):
#     imgs_row = np.hstack((img_set1[c + r*num_cols],img_set2[c + r*num_cols]))
#     for c in range(1,num_cols):
#         imgs_row = np.hstack((imgs_row, img_set1[c + r*num_cols],img_set2[c + r*num_cols]))
#     starter = np.vstack((starter, imgs_row))


def create_img_panel(img_set1, img_set2, num_rows, num_cols):
    starter = np.hstack((img_set1[0],img_set2[0]))
    for c in range(1,num_cols):
        starter = np.hstack((starter, img_set1[c],img_set2[c]))
    c = 0
    for r in range(1,num_rows):
        imgs_row = np.hstack((img_set1[c + r*num_cols],img_set2[c + r*num_cols]))
        for c in range(1,num_cols):
            imgs_row = np.hstack((imgs_row, img_set1[c + r*num_cols],img_set2[c + r*num_cols]))
        starter = np.vstack((starter, imgs_row))
    cv2.imwrite("images/seg_good_img_test.png", starter)





bad_class_labels = labels[bad_ind,:]
good_class_labels = labels[good_ind,:]


X_train = np.vstack((good_imgs, bad_imgs))
Y_train_seg = np.vstack((seg_good_imgs, seg_bad_imgs))
Y_train_seg = np.stack((Y_train_seg, np.subtract(np.ones(Y_train_seg.shape),Y_train_seg)),-1)
# Y_train_seg = np.moveaxis(Y_train_seg, 1,-1)
Y_train_class = np.vstack((good_class_labels, bad_class_labels))

num_test_good = 500
num_test_bad = 9500

good_rand_ind = np.random.choice(good_ind, size = len(good_ind), replace= False)
bad_rand_ind = np.random.choice(bad_ind, size = len(bad_ind), replace= False)

good_test_ind = good_rand_ind[:num_test_good]
good_train_ind = good_rand_ind[num_test_good:]

bad_test_ind = bad_rand_ind[:num_test_bad]
bad_train_ind = bad_rand_ind[num_test_bad:]

test_ind = np.concatenate((good_test_ind, bad_test_ind))
X_test = X_train[test_ind]
Y_test_seg = Y_train_seg[test_ind]
Y_test_class = Y_train_class[test_ind]


train_ind = np.concatenate((good_train_ind, bad_train_ind))

X_train = X_train[train_ind]
Y_train_seg = Y_train_seg[train_ind]
Y_train_class = Y_train_class[train_ind]

np.save("data/X_train_sc.npy", X_train)
np.save("data/Y_seg_train_sc.npy", Y_train_seg)
np.save("data/Y_class_train_sc.npy", Y_train_class)

np.save("data/X_test_sc.npy", X_test)
np.save("data/Y_seg_test_sc.npy", Y_test_seg)
np.save("data/Y_class_test_sc.npy", Y_test_class)