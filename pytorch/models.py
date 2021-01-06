import torch.nn as nn
from layer_utils import fftshift2, GenConvBlock, DataConsist, ifft2, fft2

class KIKI(nn.Module):
    def __init__(self, m):
        super(KIKI, self).__init__()

        conv_blocks_K = [] 
        conv_blocks_I = []
        
        for i in range(m.iters):
            conv_blocks_K.append(GenConvBlock(m.k, m.in_ch, m.out_ch, m.fm))
            conv_blocks_I.append(GenConvBlock(m.i, m.in_ch, m.out_ch, m.fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = m.iters

    def forward(self, kspace_us, mask):        
        rec = fftshift2(kspace_us)
        
        for i in range(self.n_iter):
            rec = self.conv_blocks_K[i](rec)
#            rec = DataConsist(fftshift2(rec), kspace_us, mask, is_k = True)
            rec = fftshift2(rec)
            rec = ifft2(rec)
            rec = rec + self.conv_blocks_I[i](rec)
            rec = DataConsist(rec, kspace_us, mask)
            
            if i < self.n_iter - 1:
                rec = fftshift2(fft2(rec))
        
        return rec