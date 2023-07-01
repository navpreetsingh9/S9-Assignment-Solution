import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

dropout_value = 0.05

class Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """
    def __init__(self):
        """Initialize Network

        Args:
        """
        super(Net, self).__init__()
        # input 32/1/1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), groups=3, padding=1, bias=False),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/3

        self.upsample1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, stride=1, bias=False),
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/5

        self.diltdconvblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
        ) # output_size = 32/7


        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/9

        self.upsample2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, stride=2, bias=False),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), groups=32, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/11


        self.diltdconvblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=9, bias=False),
        ) # output_size = 16/12/2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/16/2

        self.upsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, stride=2, bias=False),
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 16/20/2

        self.diltdconvblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=5, bias=False),
        ) # output_size = 8/22/4

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8/30/4
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8/38/4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), groups=128, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 8/46/4
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1)
        ) # output_size = 1

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        """Convolution function

        Args:
            x (tensor): Input image tensor

        Returns:
            tensor: tensor of logits
        """
        x = self.convblock1(x)
        y = self.upsample1(x)
        x = self.convblock2(x)
        x = y + self.diltdconvblock1(x)
        x = self.convblock3(x)
        y = self.upsample2(x)
        x = self.convblock4(x)
        x = y + self.diltdconvblock2(x)
        x = self.convblock5(x)
        y = self.upsample3(x)
        x = self.convblock6(x)
        x = y + self.diltdconvblock3(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)        
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)         