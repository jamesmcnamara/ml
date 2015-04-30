from nose.tools import assert_almost_equal, assert_equal, assert_true, assert_false
import numpy as np

from ml.clustering.k_means import K_Means
from ml.clustering.gmm import GMM

class TestKMeans:
    def setup(self):
        data = np.array([[1, 1], [1.3, 0.9], [2,1], 
                        [1.6, 1.4], [0.4, 0.6],             # Cluster 1 
                        [10, 10], [9.9, 8.9], [11, 10.9],
                        [8.6, 9.8], [10.5, 10.4]])          # Cluster 2
        self.labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.means = np.array([[1, 1], [10, 10]])
        self.k_means = K_Means(data=data, k=2)
        self.k_means.means = np.array([[10,1], [1, 10]]) 

    def test_init(self):
        assert_equal(len(self.k_means.means), 2)
        assert_equal(len(self.k_means.assignments), len(self.k_means.data))

    def test_classify(self):
        means = np.array([[1, 1], [10, 10]])
        for row, label in zip(self.k_means.data, self.labels):
            assert_equal(self.k_means.classify(row, means), label)

    def test_update_needed(self):
        assert_false(self.k_means.update_needed(self.k_means.data, self.labels, 
                                        self.means))
        assert_true(self.k_means.update_needed(self.k_means.data, 
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 
                                       self.means))
   
    def test_update_centroids(self):
        updated = self.k_means.update_centroids(self.k_means.data, self.labels, 2)
        actuals = np.array([[1.26, 0.98], [10., 10.]])
        for mean, actual in zip(updated, actuals):
            for m_i, act_i in zip(mean, actual):
                assert_almost_equal(m_i, act_i)

    def test_cluster(self):
        assert_true(self.k_means.update_needed(self.k_means.data,
                                       self.k_means.assignments,
                                       self.k_means.means))
        self.k_means.cluster()
        assert_false(self.k_means.update_needed(self.k_means.data,
                                        self.k_means.assignments, 
                                        self.k_means.means))

    def test_sse(self):
        self.k_means.cluster()
        assert_almost_equal(7.24, self.k_means.sse)

class TestGMM:
    def setup(self):
        data = np.array([[1, 1], [1.3, 0.9], [2,1], 
                        [1.6, 1.4], [0.4, 0.6],             # Cluster 1 
                        [10, 10], [9.9, 8.9], [11, 10.9],
                        [8.6, 9.8], [10.5, 10.4]])          # Cluster 2
        self.labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.gmm = GMM(data=data, k=2, use_k_means=True, norm_range=10)

    def test_init(self):
        assert_equal(self.gmm.data.shape, self.gmm.zs.shape)
        assert_equal((len(self.gmm.data.T), len(self.gmm.data.T)), self.gmm.covs[0].shape)
        assert_equal(len(self.gmm.piks), self.gmm.k)
        assert_equal(sum(self.gmm.piks), 1)

    def test_update_zs(self):
        from code import interact
        interact(local=locals())
