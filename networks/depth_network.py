from turtle import forward
import torch
import torch.nn as nn
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder

class DepthNetwork(nn.Module):

    def __init__(self) -> None:
        super(DepthNetwork, self).__init__()

        self.encoder = ResnetEncoder(50, True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, use_skips=True)
        

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
