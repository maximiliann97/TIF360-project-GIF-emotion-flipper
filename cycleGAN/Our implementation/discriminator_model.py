import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
        nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(self.block(in_channels, feature, stride=1 if feature == features[
                -1] else 2))  # stride of 2 for all features except the last one
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )



    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))     # to make sure it's between zero or one



# def test():
#     x = torch.randn((5, 3, 256, 256))
#     model = Discriminator(in_channels=3)
#     preds = model(x)
#     print(preds.shape)
#
# if __name__ == "__main__":
#     test()

