import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm


def pairwise_cosine(x1, x2):
    x1 = x1 / np.linalg.norm(x1, axis=1).reshape(-1, 1)
    x2 = x2 / np.linalg.norm(x2, axis=1).reshape(-1, 1)
    return 1 - x1 @ x2.T


def pairwise_euclidean(x1, x2):
    N1 = np.size(x1, axis=0)
    N2 = np.size(x2, axis=0)
    square = np.einsum('ij,ij->i', x1, x2)
    dist = square.reshape(N1, 1) + square.reshape(1, N2) - 2 * np.einsum('ij,kj->ik', x1, x2)
    return dist


def get_shuffle_idx(bs):
    shuffle_value = np.random.permutation(bs)  # index 2 value
    reverse_idx = np.zeros(bs)
    for i in range(bs):
        reverse_idx[i] = np.where(shuffle_value == i)[0]
    return shuffle_value.astype(int), reverse_idx.astype(int)


class PAC:
    """
        Probability Aggregation Clustering
        Parameters
        ----------
        m : float, default=1.01
            Weight exponent of PAC, which controls the fuzzy degree of clustering

        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.

        max_iter : int, default=300
            Maximum number of iterations of the PAC algorithm for a
            single run.

        error :  float, default=0.001
            Termination parameter of the PAC for early stop.

        seed : int  Random seed instance or None, default=None
            Determines random seed generation for probability initialization.

        uniformity : int, default=1000
            The uniformity of initialization, when uniformity is large,
            the probability initialization becomes more uniform.
        ----------
    """

    def __init__(self,
                 m=1.01,
                 n_clusters=8,
                 metric='cosine',
                 max_iter=100,
                 error=0.001,
                 random_state=None,
                 uniformity=1000):
        self.m = m
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.error = error
        self.uniformity = uniformity
        self.random_state = random_state
        self.distance_metric = {'euclidean': pairwise_euclidean, 'cosine': pairwise_cosine}[metric]

    def random_init(self, num: int, random_state=None):
        if isinstance(self.random_state, int):
            np.random.seed(random_state)
        p = np.random.randn(num, self.n_clusters) / self.uniformity
        p = np.exp(p) / np.exp(p).sum(1).reshape(num, -1)
        return p

    def predict(self, X):
        # agg_numbers = 1000
        N = np.size(X, axis=0)
        agg_sample = [i for i in range(N)]
        p = self.random_init(N, random_state=self.random_state)
        dist = self.distance_metric(X, X)
        norm = dist.sum() / self.n_clusters
        for iter in tqdm(range(self.max_iter)):
            p_0 = deepcopy(p)
            for i in range(N):
                # agg_idx = random.sample(agg_sample, agg_numbers)
                # frac = norm / np.array(np.einsum('i,ik->k', dist[agg_idx, i], p[agg_idx, :] ** self.m), dtype=object)
                frac = norm / np.array(np.einsum('i,ik->k', dist[:, i], p ** self.m), dtype=object)
                frac = frac / frac.mean()
                scores = np.power(frac, int(1 / (self.m - 1)))
                p[i, :] = (scores / (scores.sum()))
            if iter > 10 and np.sum(np.abs(p - p_0)) / N < self.error:
                # base iterations (the minimum number of iterations to be performed)
                break
        return p


    def adjacent_matrix(self, D):
        N = np.size(D, axis=0)
        p = self.random_init(N, random_state=self.random_state)
        for iter in range(self.max_iter):
            p_0 = deepcopy(p)
            for i in range(N):
                scores = 1 / (np.einsum('i,ik->k', D[:, i], p ** self.m)) ** (1 / (self.m - 1))
                p[i, :] = (scores / (scores.sum()))
            if iter > 10 and np.sum(np.abs(p - p_0)) / N < self.error:
                # base iterations (the minimum number of iterations to be performed)
                break
        return p
