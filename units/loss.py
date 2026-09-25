import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from units.unit import *


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.MSE = nn.MSELoss() # reduction = 'none'
        self.CompactLoss = ProtoFeatureLoss()
    def forward(self, input, target):
        return []

    def forward_MSE(self, x, x_hat):
        return self.MSE(x, x_hat)




    def contrastive_loss(self, z_i, z_j, temperature=0.5, tau=0.01):
        """Compute contrastive loss between positive pairs and negative samples"""
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()

        z = torch.cat([z_i, z_j], dim=0)


        sim_matrix = torch.matmul(z, z.T) / temperature
        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)

        pos_sim = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        neg_sim = sim_matrix[mask].reshape(N, -1)


        labels = torch.zeros(N).to(pos_sim.device).long()
        logits = torch.cat((pos_sim, neg_sim), dim=1)
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(logits, labels)

        return loss / N * tau

    def get_sim(self,centroids):

        structure = torch.mm(centroids, centroids.T)


        return F.softmax(structure, dim=1)

