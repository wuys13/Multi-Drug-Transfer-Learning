from types_ import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from molecules import Molecules
from neural_fingerprint import NeuralFingerprint



class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, normalize_flag=False):
        super(EncoderDecoder_1, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag


    def forward(self, smiles ,input: Tensor) -> Tensor:
        encoded_cell = self.encode(input)
        if self.normalize_flag:
            encoded_cell = nn.functional.normalize(encoded_cell, p=2, dim=1)
            encoded_smiles = nn.functional.normalize(smiles, p=2, dim=1)
        output = self.decoder(torch.cat((encoded_cell, encoded_smiles), dim=-1))

        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

