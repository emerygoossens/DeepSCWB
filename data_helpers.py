import numpy as np
import cv2
import os

def load_data(imagedirs, labelfile):
    label = np.loadtxt(labelfile, dtype = 'str', delimiter = ",", skiprows = 1)
    # load the data & labels as a list and save
    learning_rate = 0.5
    images = []
    labels = []
    filenames = []
    j = 0 
    indices = []
    for root, dirs, files in os.walk(imagedirs):
        for name in files:
            filename = os.path.join(root, name)
            # skip non .tiff files
            if filename.endswith(".tiff"):
                filenames.append(filename)
                indices.append(str(j))
                j += 1
                # store the image into the data list
                images.append(cv2.imread(filename, -1).ravel())
                # store the label as "one-hot vectors" (see tensorflow tutorial)
                if (int(label[np.where(name == label[:,0])[0][0],1]) == 0):
                    labels.append(np.array([1,0]))
                else:
                    labels.append(np.array([0,1]))
    ### convert list to a numpy array and then return files
    return np.vstack(images), np.vstack(labels), np.vstack(filenames), np.vstack(indices)

#~ End load_data()

# flip_type must be a tuple... single element tuples take the form (#, ); e.g. (1, )
def flip_training_data(train_images, train_labels, flip_type = (1,2,3)):
    # empty list to store flipped images and labels
    flipped_images, flipped_labels = [], []

    # cycle through the training images and flip them...
    # TODO: we can probably get some additional speed up here...
    for i in np.arange(train_images.shape[0]):
        temp_image = train_images[i, :].reshape(200, 50)

        # horizontal flip
        if 1 in flip_type:
            flipped_images.append(np.fliplr(temp_image).ravel())
            flipped_labels.append(train_labels[i, :])

        # vertical flip
        if 2 in flip_type:
            flipped_images.append(np.flipud(temp_image).ravel())
            flipped_labels.append(train_labels[i, :])

        if 3 in flip_type:
            flipped_images.append(np.fliplr(np.flipud(temp_image)).ravel())
            flipped_labels.append(train_labels[i, :])
    # End for loop

    # convert to numpy array and append to training_sets
    flipped_images = np.vstack(flipped_images)
    flipped_labels = np.vstack(flipped_labels)

    train_images = np.append(train_images, flipped_images, axis = 0)
    train_labels = np.append(train_labels, flipped_labels, axis = 0)

    return train_images, train_labels
# End flip_training_data()

# TODO: generator functions have memory... can this apply here?
# def get_next_batch(train_images, train_labels, batch_index, bsize):
#     # probably not as sexy as Google's implmentation, but neither are you...
#     batch_xs = train_images[batch_index:(batch_index + bsize), :]
#     batch_ys = train_labels[batch_index:(batch_index + bsize), :]
#     batch_index += bsize
#
#     return batch_xs, batch_ys, batch_index
# #~ End get_next_batch()
