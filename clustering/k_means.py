from operator import ne as not_equal

import numpy as np

from ml.utils.normalization import normalize

class K_Means:
    def __init__(self, data=None, k=0, iterations=1e4, norm_range=2):
        self.norm_range = norm_range
        self.data = (normalize("arithmetic", data) * norm_range) - (norm_range/2)
        self.k = k
        dim = len(self.data.T)
        self.means = np.random.uniform(low=-norm_range/2, high=norm_range/2, size=(k, dim))
        self.assignments = np.zeros(len(self.data), dtype="int32")
        self.converged = False
        self.iterations = 0
        self.max_iterations = iterations
    
    @property
    def sse(self):
        return self.sse_static(self.data, self.assignments, self.means, self.norm_range)
    
    def cluster(self):
        try:
            self.means = self.cluster_static(self.data, self.assignments, self.means)
            return True
        except ClusterException:
            self.means = np.random.uniform(low=-self.norm_range/2, high=self.norm_range/2, size=self.means.shape)
            self.iterations += 1
            return False

    @staticmethod
    def cluster_static(data, assignments, means):
        while K_Means.update_needed(data, assignments, means):
            for i, row in enumerate(data):
                assignments[i] = K_Means.classify(row, means)
            means = K_Means.update_centroids(data, assignments, len(means))
        return means

    @staticmethod
    def classify(row, means):
        return np.argmin(np.array([np.linalg.norm(row - mean)**2 
                                   for mean in means]))

    @staticmethod
    def update_needed(data, assignments, means):
        return any(map(not_equal, assignments, 
                                 (K_Means.classify(row, means) 
                                 for row in data)))
    
    @staticmethod
    def update_centroids(data, assignments, k):
        clusters = [[] for _ in range(k)]
        for row, assigned in zip(data, assignments):
            clusters[assigned].append(row)

        if not all(clusters): 
            raise ClusterException
        
        return np.array([np.mean(np.array(cluster), axis=0) for cluster in clusters])

    @staticmethod
    def sse_static(data, assignments, means, norm_range):
        return sum(np.linalg.norm(row - means[label])**2 / norm_range**2
                   for row, label in zip(data, assignments))

class ClusterException(Exception):
    pass
