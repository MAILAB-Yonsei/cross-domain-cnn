import argparse

def GenConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
    parser.add_argument('--model_name', type=str, default='KIKI', help='model name')
    parser.add_argument('--dataset_name', type=str, default='HCP_MGH_T1w', help='name of the dataset')
    parser.add_argument('--acc_rate', type=float, default=5.02, help='acceleration ratio')
    parser.add_argument('--acs_num', type=int, default=16, help='the number of acs lines')
    parser.add_argument('--mask_name', type=str, default='', help='mask name')
    
    parser.add_argument('--gpu_alloc', type=int, default=3, help='GPU allocation (0 to 3)')
    
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate') # start: 0.00005
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    
    parser.add_argument('--im_height', type=int, default=256, help='size of image height')
    parser.add_argument('--im_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=2, help='number of image channels')
    
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
    parser.add_argument('--mask_index', type=int, default=0, help='validation/test mask index')
    
    parser.add_argument('--data_augment', type=bool, default=False, help='32-fold data augmentation')
    parser.add_argument('--random_sampling', type=bool, default=False, help='Generate random sampling patterns during training')
    
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--i', type=int, default=25)
    parser.add_argument('--iters', type=int, default=2)    
    parser.add_argument('--fm', type=int, default=64)
    # parser.add_argument('--dota_lines', type=int, default=16)
    opt = parser.parse_args()
    
    parser.add_argument('--in_ch',  type=int, default=opt.channels)
    parser.add_argument('--out_ch', type=int, default=opt.channels)
    opt = parser.parse_args()
    
    print(opt)
    
    return opt
