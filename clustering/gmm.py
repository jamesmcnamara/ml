from math import log, exp
from itertools import count

import numpy as np
from scipy.stats import multivariate_normal

from ml.clustering.k_means import K_Means
from ml.utils.normalization import normalize

class GMM:
    def __init__(self, data=None, k=0, threshold=1e-2, norm_range=2, use_k_means=False):
        self.data = (normalize("arithmetic", data) * norm_range) - (norm_range/2)
        self.norm_range = norm_range
        self.k = k
        self.assignments = np.zeros(len(self.data))
        self.zs = np.random.uniform(size=(len(data), k))
        for i, row in enumerate(self.zs):
            self.zs[i] /= row.sum()
        if use_k_means:
            k_means = K_Means(data=data, k=k, norm_range=norm_range)
            cluster_success = k_means.cluster()
            while not cluster_success:
                cluster_success = k_means.cluster()
            print("k_means clustered")
            self.mus = k_means.means
            self.covs = np.array([np.cov(self.data[k_means.assignments == i].T) for i in range(k)])
            self.piks = np.array([sum(k_means.assignments == i) / len(data) for i in range(k)])
        else:
            dim = len(self.data.T)
            self.mus = np.random.uniform(high=norm_range/2, low=-(norm_range/2), size=(k, dim))
            cov = np.cov(self.data.T)
            self.covs = np.array([cov for _ in range(k)])
            self.piks = np.zeros(k)
            self.piks.fill(1/k)
        self.threshold = threshold
        self.old_error = 1e99
        self.current_error = 1e99

    def classify(self):
        for i in range(len(self.assignments)):
            self.assignments[i] = np.argmax(self.zs[i])

    @property
    def sse(self):
        return sum(np.linalg.norm(row - self.mus[self.assignments[i]])**2
                   for i, row in enumerate(self.data))

    def cluster(self):
        while not self.converged():
            self.update_gaussians()
            self.update_zs()

    def update_zs(self):
        gaussians = [multivariate_normal(mu, cov, allow_singular=True) 
                     for mu, cov in zip(self.mus, self.covs)]
        for i, row in enumerate(self.data):
            #print("pdfs: {}".format([gj.pdf(row) for gj in gaussians]))
            probs = [pj * gj.pdf(row) for pj, gj in zip(self.piks, gaussians)]
            denominator = sum(probs)
            self.zs[i] = [pr / denominator for pr in probs]
        for z in self.zs:
            assert 0.9 < z.sum() < 1.1, "The sum is {:.3f}".format(z.sum())

    def update_gaussians(self):
        n = self.zs.sum(axis=0)
        for k in range(self.k):
            self.mus[k] = sum((zi[k] * row) / n[k] 
                              for zi, row in zip(self.zs, self.data))
            self.covs[k] = (1 / n[k]) * sum(zi[k] * np.outer(row-self.mus[k], (row-self.mus[k]).T)
                                            for zi, row in zip(self.zs, self.data))
            self.piks[k] = n[k] / n.sum()

    def pdf(self, x, mu, cov, k):
        try:
            inverse = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(cov)
            det = 1/ (np.linalg.det(inverse) + 1e-13)
        else:   
            det = np.linalg.det(cov)
            print("real det")
        normalizing = (2 * np.pi)**(k/2) * np.sqrt(det)
        exponent = -0.5 * np.dot(np.dot(x-mu, inverse), x-mu)
        return 1 / normalizing * exp(exponent)

    def converged(self):
        self.old_error = self.current_error
        gaussians = [multivariate_normal(mu, cov, allow_singular=True) 
                    for mu, cov in zip(self.mus, self.covs)]
        def error(row):
            return log(sum(pij * gj.pdf(row) for pij, gj in zip(self.piks, gaussians)))
        self.current_error = sum(map(error, self.data))
        return abs(self.current_error - self.old_error) < self.threshold 
