###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# and
# https://github.com/neoamos/3d-pix2pix-CycleGAN/blob/985a98dad8ed55118357932ac9076e5a4040f7dd/data/image_folder.py#L24
# Modified og code for general purpose 3D image loading (with h5py lib)
###############################################################################



from PIL import Image
import os
import os.path
# For file type debugging:
import re

def get_filetype(filename):
    return re.split("_.", filename)[1]


def make_general_dataset(dir):
    data = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # Warning: there's no check to make sure the dataset contains only appropriate filetypes
    # List of file_types allows easier debugging  
    file_types = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            file_type = get_filetype(fname)
            if file_type not in file_types:
                file_types.append(file_type)
            path = os.path.join(root, fname)
            data.append(path)

    print("File types read in: ", file_types)
    return data





