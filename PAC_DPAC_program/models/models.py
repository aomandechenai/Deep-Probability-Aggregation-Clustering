import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_cosine(x1, x2):
    x1 = x1 / torch.linalg.norm(x1, axis=1).reshape(-1, 1)
    x2 = x2 / torch.linalg.norm(x2, axis=1).reshape(-1, 1)
    return 1 - x1 @ x2.T


def pairwise_euclidean(x1, x2):
    N1 = x1.shape[0]
    N2 = x2.shape[0]
    square = torch.einsum('ij,ij->i', x1, x2)
    dist = square.reshape(N1, 1) + square.reshape(1, N2) - 2 * torch.einsum('ij,kj->ik', x1, x2)
    return dist


class SimCLR(nn.Module):
    def __init__(self, resnet, dim_in, class_dim, feature_dim=128):
        super(SimCLR, self).__init__()
        self.resnet = resnet
        self.dim_in = dim_in
        self.feature_dim = feature_dim
        self.cluster_num = class_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.feature_dim)
        )

    def forward(self, img):
        x = self.resnet(img)
        z = self.mlp(x)
        z = F.normalize(z, dim=1)
        return z


class Network(nn.Module):
    def __init__(self, resnet, dim_in, class_dim, feature_dim=128):
        super(Network, self).__init__()
        self.resnet = resnet
        self.dim_in = dim_in
        self.feature_dim = feature_dim
        self.cluster_num = class_dim

        self.projection_head = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.feature_dim)
        )
        self.self_labeling_head = nn.Linear(self.dim_in, self.cluster_num)

        self.cluster_head = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.dim_in),
            nn.BatchNorm1d(self.dim_in),
            nn.ReLU(),
            nn.Linear(self.dim_in, self.cluster_num)
        )


    def forward(self, img):
        N = int((img.size(0)) / 2)
        x = self.resnet(img)
        z = self.projection_head(x)
        z = F.normalize(z, dim=1)
        p = self.cluster_head(x)
        u = self.self_labeling_head(x)
        return z, p[:N, :], u[N:, :]

    def PAC_online(self, img, m=1.03):
        N = int((img.size(0)))
        with torch.no_grad():
            """ online PAC program, which is the one iteration of PAC program"""
            x = self.resnet(img)
            p = self.cluster_head(x)
            q = F.softmax(p, dim=1)
            D = pairwise_cosine(x, x)
            D[range(N), range(N)] = 0
            G = torch.matmul(D, q).double()
            scores = (1 / G) ** (1 / (m - 1))
            q = (scores / (scores.sum(1)).view(-1, 1))
        return q, p

    def test_forward(self, img):
        with torch.no_grad():
            x = self.resnet(img)
            p = self.cluster_head(x)
            x = F.normalize(x, dim=1)
        return x, F.softmax(p, dim=1)