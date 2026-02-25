import torch
import torch.nn as nn

class DeblurGANv2Generator(nn.Module):
    """Lightweight DeblurGAN-v2 Generator for image deblurring"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # Input conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),     # Middle conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),      # Output conv
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)