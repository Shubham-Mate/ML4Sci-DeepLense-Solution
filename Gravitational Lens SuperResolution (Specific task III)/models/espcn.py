import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    def __init__(self, kernel_size, inp_channels=1, out_channels=1):
        super(ResBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=inp_channels, 
            out_channels=out_channels, 
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            padding_mode='reflect'
            )
        
        self.residual_conv = torch.nn.Conv2d(
            in_channels=inp_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = out + self.residual_conv(x)
        return out
    

class ESPCN(torch.nn.Module):
    def __init__(self, kernel_sizes, inp_channels=None, out_channels=None, scale_factor=1, final_kernel_size=5):
        super(ESPCN, self).__init__()
        self.inp_channels = [1 for i in range(kernel_sizes)] if inp_channels == None else inp_channels
        self.out_channels = [1 for i in range(kernel_sizes)] if out_channels == None else out_channels

        self.res_blocks = torch.nn.ModuleList([ResBlock(kernel_size, inp_channel, out_channel)
                           for kernel_size, inp_channel, out_channel in zip(kernel_sizes, self.inp_channels, self.out_channels)])
        
        self.initial_layers = torch.nn.Sequential(*self.res_blocks)
        self.upscale_layer = torch.nn.Conv2d(
            in_channels=out_channels[-1],
            out_channels=scale_factor**2,
            kernel_size=final_kernel_size,
            padding='same',
            padding_mode='reflect'
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.upscale_layer(x)
        out = self.pixel_shuffle(x)
        return out