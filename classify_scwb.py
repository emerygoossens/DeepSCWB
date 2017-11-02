# import tensorflow as tf
import sys
import os
os.chdir('/noback/scwb_shared/code')

import time

start = time.time()

import skimage.io as io
import numpy as np
import cv2
import nms
from keras.models import load_model

os.chdir('/noback/scwb_shared/')

# global variables
patch_size = 16
depth = 50
num_hidden = 128
image_size_1 = 200
image_size_2 = 50
num_labels = 2
num_channels = 1 

saved_model_dir = "saved_models/"
filename_to_classify = "full_images/MCF7-GFP_aGFP_btub_g400_p80_20160612_EMERY_635.tif"
rowstride = 90
colstride = 5
prob_cutoff = 0.999
overlap = .2




set_seed = 6
batch_multiplier = 1

batch_size = 100000

model_id = "5370D4"
model_name = "//noback/scwb_shared/saved_models/model_{}/model_{}".format(model_id, model_id)

learning_rate = 5e-6

validation_set_multiplier = 16/batch_multiplier
test_set_multiplier = 16/batch_size

reload_data = True
save_coords = True
run_classifier = True
print("Model name:" + model_name)

""" IMAGE PROCESSING STEPS """

print "[INFO] Loading image: {}".format(filename_to_classify.split("/")[-1])
image = io.imread(filename_to_classify, as_grey = True)
if image.shape[0] > image.shape[1]:
    image = np.rot90(image)

# convert image to 8-bit
image = image.astype(np.float32) / 256.0

print "[INFO] Preparing the image for classification."

# pad the image and take the interior without the padding
image = nms.padImage(img = image, winHeight = image_size_1, winWidth = image_size_2, stride = 5)
image = image[:, :, np.newaxis]

# create the grid of test images
rowdim = np.arange(0, image.shape[0] - image_size_1 - 5 + rowstride - 1, step = rowstride)
coldim = np.arange(0, image.shape[1] - image_size_2  - 5 + colstride - 1, step = colstride)

row_coords, col_coords = np.meshgrid(rowdim, coldim)
sample_coords = np.hstack((row_coords.ravel().reshape(-1, 1),
                           col_coords.ravel().reshape(-1, 1)))

garbage_upper_ind = np.where(sample_coords[:,0] > (image.shape[0] -600))[0]
garbage_lower_ind = np.where(sample_coords[:,0] <500)[0]
garbage_ind = np.hstack((garbage_upper_ind, garbage_lower_ind))

garbage_right_ind = np.where(sample_coords[:,1] >(image.shape[1] -150))[0]
garbage_ind = np.hstack((garbage_ind, garbage_right_ind))

garbage_left_ind = np.where(sample_coords[:,1] <150)[0]
garbage_ind = np.hstack((garbage_ind, garbage_left_ind))

sample_coords = np.delete(sample_coords, garbage_ind, axis = 0)


new_image = image.reshape(image.shape[0], image.shape[1])



img8 = np.zeros((new_image.shape[0], new_image.shape[1], 3), dtype = np.uint8)
img8[:,:,0] = new_image.astype('uint8')
img8[:,:,1] = new_image.astype('uint8')
img8[:,:,2] = new_image.astype('uint8')

for i in range(sample_coords.shape[0]):
    cv2.rectangle(img = img8,
                 pt1 = (sample_coords[i, 1],
                        sample_coords[i, 0]),
                 pt2 = (sample_coords[i, 1] + image_size_2,
                        sample_coords[i, 0] + image_size_1),
                 color = (0, 255, 0),
                 thickness = 2)

r = 600.0 / img8.shape[0]
dim = (int(img8.shape[1] * r), 600)

row_starter = 1750
col_starter = 1800
figure_output = 255 - new_image[row_starter:row_starter + 400, col_starter: col_starter + 600 ]


cv2.imwrite("classified_images/figure2_B.jpg", cv2.resize(figure_output, dim, interpolation = cv2.INTER_AREA))


cv2.imwrite("classified_images/unclassified_image_all.jpg", cv2.resize(img8, dim, interpolation = cv2.INTER_AREA))


# for j in range(sample_coords.shape[0]):
#     sample_coords[j,:]  = sample_coords[j,:] + j%5
if reload_data:
    test_data = np.zeros((sample_coords.shape[0], image_size_1, image_size_2, num_channels))
    for i in range(sample_coords.shape[0]):
        rowstart = sample_coords[i, 0]
        rowend = rowstart + image_size_1
        colstart = sample_coords[i, 1]
        colend = colstart + image_size_2
        test_data[i, ...] = image[rowstart:rowend,colstart:colend]
    np.save("data/full_image_data.npy",test_data)
else:
    test_data = np.load("data/full_image_data.npy")


model = load_model(model_name)
print("Using model: {}".format(model_name))


if run_classifier:
    class_probs = np.zeros((test_data.shape[0], 2), dtype = float)
    for i in range(max(test_data.shape[0]/batch_size,1)):
            print("Computing Probabilities: {}".format(i))
            temp_data = test_data[i*batch_size:(i+1)*batch_size, :,:,:].astype(np.float32)
            class_probs[i*batch_size:(i+1)*batch_size, :] = model.predict(temp_data)
    if (i+1)*batch_size <test_data.shape[0]:
        i += 1
        temp_data = test_data[i*batch_size:, :,:,:].astype(np.float32)
        class_probs[i*batch_size:, :] = model.predict(temp_data)
    np.save("data/class_probs.npy", class_probs)
else:
    class_probs = np.load("data/class_probs.npy")

print(class_probs[1:20,:])
print(class_probs[-20:,:])


# class_probs = np.delete(class_probs, garbage_ind, axis = 0)
# ind = np.argpartition(class_probs[:,0], -10000)[-10000:]
# construct the bounding boxes
positive_idx = np.where(class_probs[:, 0] <= .5)[0]
positive_coords = sample_coords[positive_idx,:]
if save_coords:
  np.save("data/positive_coords.npy", positive_coords)

scwb_boxes = np.hstack((positive_coords, np.array(positive_coords[:, 0:1] + image_size_1), np.array(positive_coords[:, 1:] + image_size_2)))

np.save("data/scwb_boxes.npy", scwb_boxes)
print(scwb_boxes)
nms_boxes = nms.non_max_suppression_fast(scwb_boxes, overlap)
np.save("data/nms_boxes.npy", nms_boxes)

print(nms_boxes)

img8 = np.zeros((new_image.shape[0], new_image.shape[1], 3), dtype = np.uint8)
img8[:,:,0] = new_image.astype('uint8')
img8[:,:,1] = new_image.astype('uint8')
img8[:,:,2] = new_image.astype('uint8')

#plot the bounding boxes...
for i in range(positive_coords.shape[0]):
    cv2.rectangle(img = img8,
                 pt1 = (positive_coords[i, 1],
                        positive_coords[i, 0]),
                 pt2 = (positive_coords[i, 1] + image_size_2,
                        positive_coords[i, 0] + image_size_1),
                 color = (0, 255, 0),
                 thickness = 2)

# resize the image...
r = 600.0 / img8.shape[0]
dim = (int(img8.shape[1] * r), 600)
cv2.imwrite("classified_images/classified_image_all_{}.jpg".format(model_id), cv2.resize(img8, dim, interpolation = cv2.INTER_AREA))


img8 = np.zeros((new_image.shape[0], new_image.shape[1], 3), dtype = np.uint8)
img8[:,:,0] = 255 - new_image.astype('uint8')
img8[:,:,1] = 255 - new_image.astype('uint8')
img8[:,:,2] = 255 - new_image.astype('uint8')


for i in range(len(nms_boxes)):
    cv2.rectangle(img = img8,
                 pt1 = (nms_boxes[i, 1],
                        nms_boxes[i, 0]),
                 pt2 = (nms_boxes[i, 3],
                        nms_boxes[i, 2]),
                 color = (0, 255, 0),
                 thickness = 2)

# resize the image...
r = 600.0 / img8.shape[0]
dim = (int(img8.shape[1] * r), 600)

cv2.imwrite("classified_images/classified_image_nms_{}.jpg".format(model_id), cv2.resize(img8, dim, interpolation = cv2.INTER_AREA))


image_fig2C = img8[row_starter:row_starter + 400, col_starter:col_starter + 600]
cv2.imwrite("classified_images/fig_2C_{}.jpg".format(model_id), cv2.resize(image_fig2C, dim, interpolation = cv2.INTER_AREA))


end = time.time()
print(end - start)