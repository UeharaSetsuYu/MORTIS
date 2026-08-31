import os
import torch
import torch.nn as nn
from torch.functional import F
import argparse
import copy
import math
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from scipy.linalg import orth
from torch.nn.utils import spectral_norm
'''自定义库 '''
from units.config import *
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from units.unit import split_complete_incomplete
import warnings
warnings.filterwarnings("ignore")


class Encoder(nn.Module):
    def __init__(self, auto_dim, activa = 'relu', batchnorm = True, LayerNorm = False, dp = 0.5):
        super(Encoder, self).__init__()
        self._dim = len(auto_dim) - 1
        self._activation = activa
        self.batchnorm = batchnorm
        self.LayerNorm = LayerNorm
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(nn.Linear(auto_dim[i], auto_dim[i + 1]))
            if i < self._dim - 1:
                if self.batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))
                elif self.LayerNorm:
                    encoder_layers.append(nn.LayerNorm(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))
                if self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                elif self._activation == 'elu':
                    encoder_layers.append(nn.ELU())
                elif self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'LeakyRelu':
                    encoder_layers.append(nn.LeakyReLU())
                elif self._activation == 'GELU':
                    encoder_layers.append(nn.GELU())
                else:
                    raise ValueError('Activation function is not supported')

        self.encoder_c = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder_c(x)



class Decoder(nn.Module):
    def __init__(self, auto_dim, activa='relu', batchnorm=True, LayerNorm=False, dp = 0.5):
        super(Decoder, self).__init__()
        self._dim = len(auto_dim) - 1
        self._activation = activa
        self.batchnorm = batchnorm
        self.LayerNorm = LayerNorm
        decoder_layers = []
        decoder_dim = [i for i in reversed(auto_dim)]
        for i in range(self._dim):
            if i == 0:
                decoder_layers.append(nn.Linear(decoder_dim[i] * 1, decoder_dim[i + 1]))
            else:
                decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))

            if i < self._dim - 1:
                if self.batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                    decoder_layers.append(nn.Dropout(dp))
                elif self.LayerNorm:
                    decoder_layers.append(nn.LayerNorm(decoder_dim[i + 1]))
                    decoder_layers.append(nn.Dropout(dp))
                if self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                elif self._activation == 'elu':
                    decoder_layers.append(nn.ELU())
                elif self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'LeakyRelu':
                    decoder_layers.append(nn.LeakyReLU())
                else:
                    raise ValueError('Activation function is not supported')

        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, x):
        return self.decoder(x)

class PrototypeGenerator(nn.Module):
    def __init__(self, feature_dim, num_clusters, hidden_dim=128, num_heads=4):
        super().__init__()
        self.num_clusters = num_clusters

        self.proto_queries = nn.Parameter(
            torch.randn(num_clusters, feature_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, Z):
        # Z: [N, D]
        Z = Z.unsqueeze(0)  # [1, N, D]

        queries = self.proto_queries.unsqueeze(0)  # [1, K, D]

        Q, A = self.attn(
            query=queries,
            key=Z,
            value=Z,
            need_weights=True
        )

        Q = Q.squeeze(0)  # [K, D]
        Q = self.ffn(Q) + Q

        return Q, A.squeeze(0)  # Q: [K, D], A: [K, N]




class Discriminator(nn.Module):
    def __init__(self, z_dim, view_num):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(z_dim, z_dim // 2),
            # nn.LeakyReLU(0.2),
            # nn.GELU(),
            nn.ReLU(),
            spectral_norm(nn.Linear(z_dim // 2, view_num)),

        )


    def forward(self, x):
        pred = self.model(x)

        return pred

class ViewGate(nn.Module):
    def __init__(self, feature_dim, view_num, hidden_dim=128, temperature=1.0):
        super().__init__()
        self.view_num = view_num
        self.temperature = temperature
        self.view_embed = nn.Embedding(view_num, feature_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 初始化为接近平均融合，训练一开始不破坏你当前的 C_mix baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, C_list, mask):
        # C_list: list of [N, D]
        # mask: list of [N, 1] or [N]
        C = torch.stack(C_list, dim=0)              # [V, N, D]
        V, N, D = C.shape

        mask = torch.stack([m.squeeze().bool() for m in mask], dim=0)  # [V, N]

        view_ids = torch.arange(V, device=C.device)
        view_emb = self.view_embed(view_ids)[:, None, :].expand(V, N, D)

        gate_input = torch.cat([C, view_emb], dim=-1)    # [V, N, 2D]
        logits = self.net(gate_input).squeeze(-1)        # [V, N]
        logits = logits / self.temperature

        logits = logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=0)             # [V, N], over views

        C_mix = (alpha.unsqueeze(-1) * C).sum(dim=0)     # [N, D]
        return C_mix, alpha
class ConExtractor(nn.Module):
    def __init__(self, dim, hidden_dim = 256, dp=0.2):
        super(ConExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        output = self.mlp(x) + x
        return output






class CausalMVC(nn.Module):
    def __init__(self, config, auto_dim, device, dp = 0.2):
        super(CausalMVC, self).__init__()
        ''' Datasets parameters '''
        self.view_num = config['view_num']
        self.input_dim = config['batch_size']
        self.cluster_num = config['class_num']
        self.device = device
        self.feature_dim = auto_dim[0][-1]
        self.dp = dp
        ''' Module '''
        self.encoder = nn.ModuleList([Encoder(dim, activa = 'relu', batchnorm = False, LayerNorm = True, dp=self.dp) for dim in auto_dim])
        self.decoder = nn.ModuleList([Decoder(dim, activa = 'relu', batchnorm = False, LayerNorm = False, dp=self.dp) for dim in auto_dim])

        self.cluster_layer_v = nn.ParameterList([
            nn.Parameter(torch.empty(self.cluster_num, self.feature_dim))
            for _ in range(self.view_num)
                 ])
        for proto in self.cluster_layer_v:
            nn.init.xavier_uniform_(proto)

        self.Con = ConExtractor(self.feature_dim)

    def forward(self, x, mask = None):

        latent_list = [self.encoder[view](x[view]) for view in range(self.view_num)]

        x_hat_list = [self.decoder[view](latent_list[view]) for view in range(self.view_num)]

        return latent_list, x_hat_list




if __name__ == '__main__':
    x = [[10, 128, 128], [20, 128, 128]]
    config = {'view_num': 2, 'batch_size': 256, 'class_num': 2, 'device': 'cuda'}
    model = CausalMVC(config, x, config['device'])
    input = [torch.randn(256, 10), torch.randn(256, 20)]
    mask = [torch.randint(0, 2, (256, 1)), torch.randint(0, 2, (256, 1))]
    print(model(input, mask))


