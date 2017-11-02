
import numpy as np 
import os
from scipy.ndimage.interpolation import rotate
import random
from autoencoder_helpers import *
from scipy import signal




data_points = 1000000
pix_to_roll = 15


all_data = reformat(np.load("data/image_dataset_aug.npy"))[:data_points,]
labels = np.load("data/image_labels_aug.npy")[:data_points,]

all_data = (all_data + 32768.0)*(256.0/65535)


XY_train = np.zeros((all_data.shape[0], 200,64,1))

XY_train[:,:,7:57,:] = all_data


np.save("data/XY_train_unsuper.npy", XY_train)





