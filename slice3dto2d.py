# Turning the 3d slices into 2d 
import h5py
import numpy as np
import os.path
import random
import matplotlib.pyplot as plt
import math

def get_slices(path_in, path_out, extension):
    
    image_3d = os.listdir(path_in)
    idx = 1
    file_list = []

    for image in image_3d:
        if image.endswith(extension):
            file_list.append(image)
    
    for image in file_list:
        # create training samples
        img_path = os.path.join(path_in, str(image))
        with h5py.File(img_path, 'r') as hf:
            img = np.array(hf['data'])

        img_shape = img.shape

        for channel in range(img_shape[2]):
            if len(img_shape) == 3:
                img_slice = img[:,:,channel]
            elif len(img_shape) == 4:
                img_slice = img[:,:,channel,:]
            
            name_out = "img_" + str(idx)
            save_samples = os.path.join(path_out, name_out)
            np.save(save_samples, img_slice)
            idx += 1





if __name__=="__main__":
    path_in = 'train_A'
    path_out = '2D_train_A_im'
    extension = '.im'
    get_slices(path_in,path_out, extension)

