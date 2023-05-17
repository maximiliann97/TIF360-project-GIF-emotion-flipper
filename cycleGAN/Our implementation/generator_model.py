import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=54, num_residuals=9):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        # Downsample Blocks
        self.down_blocks = nn.ModuleList()
        in_features = num_features
        out_features = num_features * 2
        for _ in range(2):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features
            out_features *= 2

        # Residual Blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residuals):
            self.residual_blocks.append(nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True)
            ))

        # Upsample Blocks
        self.up_blocks = nn.ModuleList()
        out_features = in_features // 2
        for _ in range(2):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features
            out_features = out_features // 2

        self.last = nn.Sequential(
            nn.Conv2d(in_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)

        # Downsample
        #skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            #skip_connections.append(x)

        # Residual blocks
        for residual_block in self.residual_blocks:
            x = x + residual_block(x)

        # Upsample
        for up_block in self.up_blocks:
            x = up_block(x)
            #x = torch.cat((x, skip_connections[-idx - 1]), dim=1)

        return self.last(x)


# def test():
#     img_channels = 3
#     img_size = 256
#     x = torch.randn((2, img_channels, img_size, img_size))
#     gen = Generator(img_channels, 9)
#     print(gen(x).shape)
#
#
# if __name__ == "__main__":
#     test()