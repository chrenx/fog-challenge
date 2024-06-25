import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the autoencoder model
class UNet(nn.Module):
    def __init__(self, size, channel):
        super(UNet, self).__init__()
        
        #* Encoding layers *************************************************************************
        self.e11 = nn.Conv1d(channel, 32, kernel_size=3, padding='same')
        self.e12 = nn.Conv1d(32, 32, kernel_size=3, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2) # 1024
        
        self.e21 = nn.Conv1d(32, 64, kernel_size=3, padding='same')
        self.e22 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2) # 512
        
        self.e31 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.e32 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.maxpool3 = nn.MaxPool1d(kernel_size=2) # 256
        
        self.e41 = nn.Conv1d(128, 256, kernel_size=3, padding='same')
        self.e42 = nn.Conv1d(256, 256, kernel_size=3, padding='same')
        self.maxpool4 = nn.MaxPool1d(kernel_size=2) # 128
        
        self.e51 = nn.Conv1d(256, 512, kernel_size=3, padding='same')
        self.e52 = nn.Conv1d(512, 512, kernel_size=3, padding='same') #128*512
        self.maxpool5 = nn.MaxPool1d(kernel_size=2) # 64

        #* Decoding layers *************************************************************************
        self.d51 = nn.Conv1d(512, 512, kernel_size=3, padding='same')
        self.d52 = nn.Conv1d(512, 512, kernel_size=3, padding='same')
        self.up5 = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2, padding=0)
        # cat: up5, e52: 512 + 512 = 1024
        
        self.d41 = nn.Conv1d(1024, 256, kernel_size=3, padding='same')
        self.d42 = nn.Conv1d(256, 256, kernel_size=3, padding='same')
        self.up4 = nn.ConvTranspose1d(256, 512, kernel_size=2, stride=2, padding=0)
        # cat: up4, e42: 512 + 256 = 768
        
        self.d31 = nn.Conv1d(768, 128, kernel_size=3, padding='same')
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.up3 = nn.ConvTranspose1d(128, 512, kernel_size=2, stride=2, padding=0)
        # cat: up3, e32: 512 + 128 = 640
        
        self.d21 = nn.Conv1d(640, 64, kernel_size=3, padding='same')
        self.d22 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.up2 = nn.ConvTranspose1d(64, 512, kernel_size=2, stride=2, padding=0)
        # cat: up2, e22: 512 + 64 = 576
        
        self.d11 = nn.Conv1d(576, 64, kernel_size=3, padding='same')
        self.d12 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.up1 = nn.ConvTranspose1d(64, 512, kernel_size=2, stride=2, padding=0)
        # cat: up1, e12: 512 + 32 = 544
        
        self.outconv = nn.Conv1d(544, 2, kernel_size=3, padding='same')

    def forward(self, x):
        #* Encoding layers *************************************************************************
        xe11 = nn.ReLU(self.e11(x))
        xe12 = nn.ReLU(self.e12(xe11))
        xp1 = self.maxpool1(xe12)
        
        xe21 = nn.ReLU(self.e21(xp1))
        xe22 = nn.ReLU(self.e22(xe21))
        xp2 = self.maxpool2(xe22)
        
        xe31 = nn.ReLU(self.e31(xp2))
        xe32 = nn.ReLU(self.e32(xe31))
        xp3 = self.maxpool3(xe32)
        
        xe41 = nn.ReLU(self.e41(xp3))
        xe42 = nn.ReLU(self.e42(xe41))
        xp4 = self.maxpool4(xe42)
        
        xe51 = nn.ReLU(self.e51(xp4))
        xe52 = nn.ReLU(self.e52(xe51))
        xp5 = self.maxpool5(xe52)
        
        #* Decoding layers *************************************************************************
        xd51 = nn.ReLU(self.d51(xp5))
        xd52 = nn.ReLU(self.d52(xd51))
        xup5 = nn.ReLU(self.up5(xd52))
        xup5 = torch.cat([xup5, xe52], dim=-1)
        
        xd41 = nn.ReLU(self.d41(xup5))
        xd42 = nn.ReLU(self.d42(xd41))
        xup4 = nn.ReLU(self.up4(xd42))
        xup4 = torch.cat([xup4, xe42], dim=-1)
        
        xd31 = nn.ReLU(self.d31(xup4))
        xd32 = nn.ReLU(self.d32(xd31))
        xup3 = nn.ReLU(self.up3(xd32))
        xup3 = torch.cat([xup3, xe32], dim=-1)
        
        xd21 = nn.ReLU(self.d21(xup3))
        xd22 = nn.ReLU(self.d22(xd21))
        xup2 = nn.ReLU(self.up2(xd22))
        xup2 = torch.cat([xup2, xe22], dim=-1)
        
        xd11 = nn.ReLU(self.d11(xup2))
        xd12 = nn.ReLU(self.d12(xd11))
        xup1 = nn.ReLU(self.up1(xd12))
        xup1 = torch.cat([xup1, xe12], dim=-1)
        
        
        return 