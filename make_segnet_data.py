
import numpy as np 
import os
from scipy.ndimage.interpolation import rotate
# os.chdir('/noback/scwb_shared/')
# import cv2

import random


from code.autoencoder_helpers import *

from scipy import signal




data_points = 1000000
pix_to_roll = 15


all_data = reformat(np.load("data/image_dataset_aug.npy"))[:data_points,]
labels = np.load("data/image_labels_aug.npy")[:data_points,]

all_data = (all_data + 32768.0)*(256.0/65535)


good_imgs = np.roll(all_data[np.where(labels[:,1] ==1)[0],:],pix_to_roll,axis = 1)

seg_good_imgs = np.zeros(good_imgs.shape)



num_images = good_imgs.shape[0]



for j in range(num_images):
    segmented_image = good_imgs[j,:,:,0]
    segmented_image = np.clip((segmented_image- np.mean(segmented_image))*100,0, 255)
    # segmented_image[:60,:] = 0
    segmented_image[-15:,:] = 0
    segmented_image[:,:10] = 0
    segmented_image[:,-10:] = 0
    segmented_image = smooth(segmented_image)
    segmented_image[np.where(segmented_image >segmented_image.mean())] = 255
    segmented_image[np.where(segmented_image <segmented_image.mean())] = 0
    seg_good_imgs[j,:,:,0] = segmented_image

seg_good_imgs = seg_good_imgs/255

seg_good_imgs_paddedz = np.zeros((seg_good_imgs.shape[0],1,200,64))
seg_good_imgs_paddedo = np.ones((seg_good_imgs.shape[0],1,200,64))
seg_good_imgs_padded = np.hstack((seg_good_imgs_paddedz,seg_good_imgs_paddedo))
# seg_good_imgs = np.reshape(seg_good_imgs,(seg_good_imgs.shape[0],200,64))
seg_good_imgs_padded[:,0,:,7:57] = seg_good_imgs[:,:,:,0]
seg_good_imgs_padded[:,1,:,7:57] = np.subtract(np.ones(seg_good_imgs.shape[:3]), seg_good_imgs[:,:,:,0])


seg_good_imgs_padded = np.reshape(seg_good_imgs_padded, (seg_good_imgs_padded.shape[0],2,64*200))
seg_good_imgs_padded = np.moveaxis(seg_good_imgs_padded, -1,1)


np.save("data/X_train_segnet.npy", np.roll(good_imgs, -pix_to_roll, axis = 1))
np.save("data/Y_train_segnet.npy", np.roll(seg_good_imgs_padded, -pix_to_roll, axis = 1))











