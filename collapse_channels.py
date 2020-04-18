# Go from segmentation mask channels to one channel with pixel intensities
# corresponding to class
import numpy as np
import h5py
import os.path
import random
import math


def get_multiclass(label):
    # For appending background channel
    background = np.zeros((label.shape[0], label.shape[1], 1))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            sum = np.sum(label[i,j,:])
            if sum == 0:
                background[i][j] = 1
            else:
                background[i][j] = 0
                
    label = np.concatenate((label, background), axis = 2)
    # label = np.reshape(label, (label.shape[0]*label.shape[1], label.shape[2]))
    
    return label


def collapse_channels(path_in, path_out, channel_pos=-1):
    """ Collapse all channels to a number of bw brightness levels
    corresponding to the number of channels """
    image_3d = os.listdir(path_in)
    extension = '.npy'

    file_list = []
    for image in image_3d:
        if image.endswith(extension):
            file_list.append(image)
    
    for image in file_list:
        # Load image
        img_path = os.path.join(path_in, str(image))
        img = np.load(img_path)

        # Append bg channel
        img = get_multiclass(img)

        img_shape = img.shape
        nc = img_shape[channel_pos]

        # collapse channels
        img = np.argmax(img, axis=-1) 
        img = np.divide(img, nc)

        # save image
        save_samples = os.path.join(path_out, str(image))
        np.save(save_samples, img)




if __name__ == '__main__':
    path_in = 'trainB'
    path_out = 'trainB_flat_ahh'
    collapse_channels(path_in, path_out)


