from types_ import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from molecules import Molecules
from neural_fingerprint import NeuralFingerprint


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, normalize_flag=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag


    def forward(self, input: Tensor) -> Tensor:
        encoded_input = self.encode(input)
        if self.normalize_flag:
            encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        output = self.decoder(encoded_input)

        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class EncoderDecoder_1(nn.Module):

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

# ZSY
class EncoderDecoder_drug(nn.Module):

    def __init__(self, encoder, decoder, drug_emb_dim, device, normalize_flag=False):
        super(EncoderDecoder_drug, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.normalize_flag = normalize_flag
        self.smiles_encoder = NeuralFingerprint(node_size=62, edge_size=6, conv_layer_sizes=[16,16], 
                                output_size=drug_emb_dim, degree_list=[0, 1, 2, 3, 4, 5], device=device)

    def convert_smile_to_feature(self,smiles, device):
        #print(smiles)
        molecules = Molecules(smiles)
        node_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('atom')]).to(device)#.double()
        edge_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('bond')]).to(device)#.double()
        return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}

    def forward(self, smiles,gex) -> Tensor:
        #input: (smiles,gex)
        encoded_cell = self.encode(gex)
        if self.normalize_flag:
            encoded_cell = nn.functional.normalize(encoded_cell, p=2, dim=1)

        encoded_smiles = self.smiles_encoder(self.convert_smile_to_feature(smiles,self.device))
        # [batch * num_node * drug_emb_dim]
        encoded_smiles = torch.sum(encoded_smiles, dim=1)
        # [batch * drug_emb_dim]

        output = self.decoder(torch.cat((encoded_cell, encoded_smiles), dim=-1))
        return output

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

