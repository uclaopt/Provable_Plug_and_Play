# header
import torch.nn as nn
from model.conv_sn_chen import conv_spectral_norm
from model.bn_sn_chen import bn_spectral_norm

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, lip=1.0, no_bn=False,
                 adaptive=False):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        if lip > 0.0:
            sigmas = [pow(lip, 1.0/num_of_layers) for _ in range(num_of_layers)]
        else:
            sigmas = [0.0 for _ in range(num_of_layers)]

        if adaptive:
            # sigmas = [5.0, 2.0, 0.68, 0.46, 0.31]
            # sigmas = [5.0, 1.0, 0.584, 0.342]
            sigmas = [5.0, 2.0, 1.0, 0.681, 0.464, 0.316]
            assert len(sigmas) == num_of_layers, "Length of SN list uncompatible with num of layers."

        def conv_layer(cin, cout, sigma):
            conv = nn.Conv2d(in_channels=cin,
                             out_channels=cout,
                             kernel_size=kernel_size,
                             padding=padding,
                             bias=False)
            if sigma > 0.0:
                return conv_spectral_norm(conv, sigma=sigma)
            else:
                return conv

        def bn_layer(n_features, sigma=1.0):
            bn = nn.BatchNorm2d(n_features)
            if sigma > 0.0:
                return bn_spectral_norm(bn, sigma=sigma)
            else:
                return bn

        layers = []
        layers.append(conv_layer(channels, features, sigmas[0]))
        layers.append(nn.ReLU(inplace=True))
        print("conv_1 with SN {}".format(sigmas[0]))

        for i in range(1, num_of_layers-1):
            layers.append(conv_layer(features, features, sigmas[i])) # conv layer
            print("conv_{} with SN {}".format(i+1, sigmas[i]))
            if not no_bn:
                layers.append(bn_layer(features, 0.0)) # bn layer
            layers.append(nn.ReLU(inplace=True))

        layers.append(conv_layer(features, channels, sigmas[-1]))
        print("conv_{} with SN {}".format(num_of_layers, sigmas[-1]))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

