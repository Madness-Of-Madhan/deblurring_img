import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),  # Memory efficient
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class MPRStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
    
    def forward(self, x):
        return torch.tanh(self.net(x))

class MPRNet(nn.Module):
    """Multi-Stage Progressive Restoration Network"""
    def __init__(self):
        super().__init__()
        self.stage1 = MPRStage()
        self.stage2 = MPRStage()
        self.stage3 = MPRStage()
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x