import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from PIL import Image
#from data.image_folder import make_dataset
import pickle
import numpy as np
# h5py helps read in .im and .seg files (& others supported):
import h5py
# For file type debugging:
import re

# following fxns from
def get_filetype(filename):
    return re.split(".", filename)[1]


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

class NoduleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.pkl_file = os.path.join(opt.dataroot, "crops.pkl")
        # self.heatmaps_dir = os.path.join(opt.dataroot, "heatmaps/real")
        # self.scans_dir = os.path.join(opt.dataroot, "scans_processed")
        # self.samples = pickle.load(open(self.pkl_file, 'rb'))
        # self.samples = [ j for i in self.samples for j in i]
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_general_dataset(self.dir_A)
        self.B_paths = make_general_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # self.transform = get_transform(opt)

        # random.shuffle(self.samples)
        
        self.scans = {}
        self.heatmaps = {}


    def __getitem__(self, index):
        # OG Author's wrote:
        # #returns samples of dimension [channels, z, x, y]

        # sample = self.samples[index]
        
        # # using A for scans and B for heatmaps (to generalize naming from what the og authors used) # written explicitly to easily change later
        # file_suffix_A = ".npz"
        # file_suffix_B = ".npz"
        
        # #load scan and heatmap if it hasnt already been loaded
        # if not self.scans.has_key(sample['suid']):
        #     print(sample['suid'])
        #     self.scans[sample['suid']] = np.load(os.path.join(self.scans_dir, sample['suid'] + file_suffix_A))['a']
        #     self.heatmaps[sample['suid']] = np.load(os.path.join(self.heatmaps_dir, sample['suid'] + file_suffix_B))['a']
        # scan = self.scans[sample['suid']]
        # heatmap = self.heatmaps[sample['suid']]

        # #crop
        # b = sample['bounds']
        # scan_crop = scan[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
        # heatmap_crop = heatmap[b[0]:b[3], b[1]:b[4], b[2]:b[5]]

        # #convert to torch tensors with dimension [channel, z, x, y]
        # scan_crop = torch.from_numpy(scan_crop[None, :])
        # heatmap_crop = torch.from_numpy(heatmap_crop[None, :])
        # return {
        #         'A' : scan_crop,
        #         'B' : heatmap_crop
        #         }

        # Generalization off og lung nodule code
        index_A = index % self.A_size
        A_path = self.A_paths[index % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # img here refers to a piece of data but does not need to be an image
# Channels in tf are in 3rd dimension and network expects 3 channels; using [Stack, Width, Height, Channels, Depth]
        with h5py.File(A_path,'r') as hf_A:
            A = np.array(hf_A['data'])
# Stack images
#            A = np.rollaxis(A, 1,3)      # swapping channel column, input of MRI images not tf compatible
#            A = np.stack((A)*3, axis=3)
#            if len(A.shape)==3:
#                A = A[:, :, :, None] * np.ones(3)[None, None, None, :]
#            print('A.shape',A.shape)
        with h5py.File(B_path,'r') as hf_B:
            B = np.array(hf_B['data'])
#            B = np.rollaxis(B, 1,3)
#            print('B.shape',B.shape)
        #convert to torch tensors with dimensions [channel, z, x, y] (from og authors)
        A = torch.from_numpy(A[None, :])
        B = torch.from_numpy(B[None, :])

        return {
                'A' : A,
                'B' : B
                }


    def __len__(self):
        return self.A_size + self.B_size

    def name(self):
        return 'NodulesDataset'

if __name__ == '__main__':
    #test
    n = NoduleDataset()
    n.initialize("datasets/nodules")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
