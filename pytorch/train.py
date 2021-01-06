import os
from math import sqrt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from layer_utils import weights_init_normal
from importlib import import_module

from datasets import ImageDataset

from config import GenConfig
from utils import GenAddPathStr, GenAugSeq, save_images

opt = GenConfig()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='%d' % opt.gpu_alloc

add_path_str = GenAddPathStr(opt)
os.makedirs('SavedModels_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)
os.makedirs('Validation_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)

GeneratorNet = getattr(import_module('models'), opt.model_name)

# Initialize generator and criterion
generator = GeneratorNet(opt)
criterion = torch.nn.MSELoss()
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()
    criterion = criterion.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)

# Optimizers
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=10**-7)

# Tensor allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_k_us_train   = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)
input_img_fs_train = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)
input_mask_train   = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)

input_k_us_valid   = Tensor(1, opt.channels, opt.im_height, opt.im_width)
input_img_fs_valid = Tensor(1, opt.channels, opt.im_height, opt.im_width)
input_mask_valid   = Tensor(1, opt.channels, opt.im_height, opt.im_width)

# Data augmentation
if opt.data_augment:
    aug_seq = GenAugSeq()
else:
    aug_seq = None

# Data loader
dataloader_train = DataLoader(ImageDataset(opt.dataset_name,
                                           aug_seq=aug_seq,
                                           acc_rate=opt.acc_rate,
                                           acs_num=opt.acs_num,
                                           random_sampling=opt.random_sampling,
                                           mask_name = opt.mask_name),
                                           batch_size=opt.batch_size, shuffle=True)

dataloader_valid = DataLoader(ImageDataset(opt.dataset_name,
                                           acc_rate=opt.acc_rate,
                                           acs_num=opt.acs_num,
                                           is_valid=True,
                                           test_num=1,
                                           mask_name = opt.mask_name),
                                           batch_size=1, shuffle=False)

    
# --------------------------
#  Training with validation
# --------------------------

if __name__ == "__main__":
    
    for i, valid_data in enumerate(dataloader_valid):
        valid_kspaces_us = Variable(input_k_us_valid.copy_(valid_data['kspace_us']))
        valid_imgs_fs    = Variable(input_img_fs_valid.copy_(valid_data['img_fs']))
        valid_mask       = Variable(input_mask_valid.copy_(valid_data['mask_rev']))
                
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, train_data in enumerate(dataloader_train):
            if train_data['kspace_us'].size(0) != opt.batch_size:
                continue
            # Training
            train_kspaces_us  = Variable(input_k_us_train.copy_(train_data['kspace_us']))
            train_imgs_fs     = Variable(input_img_fs_train.copy_(train_data['img_fs']))
            train_mask        = Variable(input_mask_train.copy_(train_data['mask_rev']))

            optimizer.zero_grad()
            train_imgs_rec = generator(train_kspaces_us, train_mask)
            loss_train     = criterion(train_imgs_rec, train_imgs_fs)
            loss_train.backward()
            optimizer.step()
            
            # Validation
            valid_imgs_rec = generator(valid_kspaces_us, valid_mask)
            loss_valid     = criterion(valid_imgs_rec, valid_imgs_fs)
            
            # Print status
            print("[Epoch %d/%d] [Batch %d/%d] [Valid loss: %.4f]" %
                  (epoch, opt.n_epochs, i, len(dataloader_train), sqrt(loss_valid.item())))
            
            # Save validation images
            batches_done = epoch * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                val_data = [valid_data['img_us'], valid_imgs_rec, valid_imgs_fs]
                save_images(val_data, loss_valid, epoch, batches_done, opt.model_name, add_path_str, Tensor)
           
        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), 'SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, epoch + 1))