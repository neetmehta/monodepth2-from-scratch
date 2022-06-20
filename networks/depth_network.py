from turtle import forward
import torch
import torch.nn as nn
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import resnet50

class DepthNetwork(nn.Module):

    def __init__(self) -> None:
        super(DepthNetwork, self).__init__()

        self.decoder = DepthDecoder([64,256,512,1024,2048])
        self.encoder = resnet50()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
