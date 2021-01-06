from glob import glob
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset
import torch

from utils import CartesianMask

class ImageDataset(Dataset):
    def __init__(self, dataset_name, is_valid=False, is_testing=False, test_num=1,
                 aug_seq=None, acc_rate=5, acs_num=0, random_sampling=False, mask_index=0, mask_name=None):
        if is_testing:
            self.files_fs = sorted(glob('../Data/%s/SR100/Test/im/*.*' % (dataset_name)))
            if test_num != -1:
                self.files_fs = self.files_fs[:test_num]
        elif is_valid:
            self.files_fs = sorted(glob('../Data/%s/SR100/Valid/im/*.*' % (dataset_name)))
            if test_num != -1:
                self.files_fs = self.files_fs[:test_num]
        else:
            self.files_fs = sorted(glob('../Data/%s/SR100/Train/im/*.*' % (dataset_name)))

        self.is_testing = is_testing
        self.aug_seq = aug_seq
        self.acc_rate = acc_rate
        self.acs_num = acs_num
        self.random_sampling = random_sampling
        self.mask_index = mask_index
        self.mask_name = mask_name

    def __getitem__(self, index):
        # 1,256,256
        img_fs = sio.loadmat(self.files_fs[index % len(self.files_fs)])['im']
        h, w = img_fs.shape
        
        # load or generate sampling mask
        if not self.random_sampling:
            if self.mask_name == None:
                mask = sio.loadmat('../../Data/SamplingMask/mask_%.2f_%d_%d/mask%d.mat' % 
                                  (self.acc_rate, self.acs_num, h, self.mask_index))['mask']
            else:
                mask = sio.loadmat('../../Data/SamplingMask/%s.mat' % self.mask_name)['mask']
        else:
            mask = CartesianMask((h, w), self.acc_rate, self.acs_num)
        
        mask_rev = np.ones_like(mask) - mask
        
        if not self.is_testing and self.aug_seq != None:
            img_fs = np.reshape(img_fs, (h, w, 1))
            img_fs = self.aug_seq.augment_image(img_fs)
            img_fs = img_fs[:,:,0]
        
        kspace_fs = np.fft.fft2(img_fs)
        kspace_us = np.fft.ifftshift(np.fft.fftshift(kspace_fs) * np.fft.fftshift(mask))
        img_us = np.fft.ifft2(kspace_us)
        
        kspace_us = kspace_us[np.newaxis]
        kspace_fs = kspace_fs[np.newaxis]
        img_us    = img_us[np.newaxis]
        img_fs    = img_fs[np.newaxis]
        mask      = mask[np.newaxis]
        
        kspace_us = np.concatenate((np.real(kspace_us), np.imag(kspace_us)), axis = 0)
        kspace_fs = np.concatenate((np.real(kspace_fs), np.imag(kspace_fs)), axis = 0)
        img_us    = np.concatenate((np.real(img_us),    np.imag(img_us)),    axis = 0)
        img_fs    = np.concatenate((np.real(img_fs),    np.imag(img_fs)),    axis = 0)
        mask      = np.concatenate((mask, mask), axis=0)
        
        kspace_us = torch.Tensor(kspace_us)
        kspace_fs = torch.Tensor(kspace_fs)
        img_us    = torch.Tensor(img_us)
        img_fs    = torch.Tensor(img_fs)
        mask      = torch.Tensor(mask)
        
        return {'kspace_us': kspace_us, 'kspace_fs': kspace_fs, 'img_us': img_us, 'img_fs': img_fs, 'mask_rev': mask_rev}

    def __len__(self):
        return len(self.files_fs)