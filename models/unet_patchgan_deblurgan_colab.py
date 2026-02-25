import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """U-Net encoder/decoder block with skip connections"""
    
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)  # Memory efficient
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)  # Memory efficient
            )
    
    def forward(self, x):
        return self.block(x)

class UNetGenerator(nn.Module):
    """DeblurGAN-style U-Net Generator with skip connections"""
    
    def __init__(self):
        super().__init__()
        # Encoder
        self.d1 = UNetBlock(3, 64)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        
        # Decoder
        self.u1 = UNetBlock(256, 128, down=False)
        self.u2 = UNetBlock(256, 64, down=False)
        
        # Output layer
        self.out = nn.ConvTranspose2d(128, 3, 4, 2, 1)
    
    def forward(self, x):
        # Encoder with skip connections
        d1 = self.d1(x)      # 64 channels
        d2 = self.d2(d1)     # 128 channels
        d3 = self.d3(d2)     # 256 channels
        
        # Decoder with skip connections
        u1 = self.u1(d3)                        # 128 channels
        u2 = self.u2(torch.cat([u1, d2], 1))   # 256->64 channels
        
        # Output
        return torch.tanh(self.out(torch.cat([u2, d1], 1)))