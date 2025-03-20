import torch
import torch.nn.functional as F


class SRCNN(torch.nn.Module):
    def __init__(self, scale_factor, n_1, ns, f_1, f_2s, f_3):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor

        self.patch_extractor = torch.nn.Conv2d(
            1, n_1, (f_1, f_1), padding="same", padding_mode="reflect"
        )

        non_linear_layers = []
        in_channels = n_1
        for n, f_2 in zip(ns, f_2s):
            non_linear_layers.append(
                torch.nn.Conv2d(
                    in_channels, n, (f_2, f_2), padding="same", padding_mode="reflect"
                )
            )
            non_linear_layers.append(torch.nn.ReLU(inplace=True))
            in_channels = n
        self.non_linear_map = torch.nn.Sequential(*non_linear_layers)

        self.reconstructor = torch.nn.Conv2d(
            ns[-1], 1, (f_3, f_3), padding="same", padding_mode="reflect"
        )

    def forward(self, x):
        upsampled_x = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")
        upsampled_x = self.patch_extractor(upsampled_x)
        upsampled_x = self.non_linear_map(upsampled_x)
        upsampled_x = self.reconstructor(upsampled_x)
        return upsampled_x
