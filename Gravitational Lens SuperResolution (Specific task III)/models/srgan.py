import torch


class GenResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GenResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="same", bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.prelu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, padding="same", bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x

        return out


class GenUpSample(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size_1, kernel_size_2, scale=1
    ):
        super(GenUpSample, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size_1, padding="same"
        )
        self.pixel_suffle = torch.nn.PixelShuffle(upscale_factor=scale)
        self.prelu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(int(out_channels / (scale**2)), 1, kernel_size_2, padding='same')

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_suffle(x)
        x = self.prelu(x)
        x = self.conv2(x)

        return x


class SRGenerator(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size_1, kernel_size_2, num_res_blocks, upscale_factor
    ):
        super(SRGenerator, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size_1, padding='same')
        self.prelu1 = torch.nn.PReLU()

        self.res_blocks = torch.nn.ModuleList(
            [
                GenResBlock(out_channels, out_channels, kernel_size_2)
                for i in range(num_res_blocks)
            ]
        )
        self.res_blocks_seq = torch.nn.Sequential(*self.res_blocks)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size_2, padding='same', bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.upscale_block = GenUpSample(
            out_channels, 256, kernel_size_1, kernel_size_2, upscale_factor
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)

        out = self.res_blocks_seq(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x

        upscaled_out = self.upscale_block(out)

        return upscaled_out


class DiscConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DiscConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, stride=2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)

        return x


class SRDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(SRDiscriminator, self).__init__()
        self.conv_blocks = torch.nn.ModuleList(
            [
                DiscConvBlock(i, o, k)
                for i, o, k in zip(in_channels, out_channels, kernel_sizes)
            ]
        )
        self.conv_blocks_seq = torch.nn.Sequential(*self.conv_blocks)
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(out_channels[-1], 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_blocks_seq(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fcn(x)

        return x
