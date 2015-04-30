from itertools import count, chain 

from nose.tools import assert_almost_equal, assert_equal
import numpy as np

from ml.classifier.perceptron import Perceptron, DualPerceptron
from ml.classifier.neural_net import NeuralNet, Node 
__author__ = 'jamesmcnamara'

and_truth_table = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])
and_results = [-1, -1, -1, 1]

class TestPerceptron:
    def test_find_ws(self):
        p = Perceptron(data=and_truth_table, labels=and_results, step=1)
        for actual, expected in zip(p.ws, [-4, 3, 2]):
            assert_equal(actual, expected)

    def test_predict(self):
        p = Perceptron(data=and_truth_table, labels=and_results, step=1)
        pred = p.predict(np.array([[0, 0], [1, 1]]))
        for p, a in zip(pred, (-1, 1)):
            assert_equal(p, a)

    def test_find_ws_dual(self):
        p = DualPerceptron(data=and_truth_table, labels=and_results, step=1)
        ws = sum(alpha * row * label for alpha, row, label in zip(p.alphas, p.data, p.labels))
        for actual, expected in zip(ws, [-4, 3, 2]):
            assert_equal(actual, expected)

    def test_predicti_dual(self):
        p = DualPerceptron(data=and_truth_table, labels=and_results, step=1)
        pred = p.predict(np.array([[0, 0], [1, 1]]))
        for p, a in zip(pred, (-1, 1)):
            assert_equal(p, a)

class TestNeuralNet:

    def test_weight_vector_size(self):
        net = NeuralNet(units=[8, 3, 8])
        for vector, exp_length in zip(net.ws, (24, 24)):
            assert_equal(len(vector), exp_length)
        net = NeuralNet(units=[8, 3, 4, 9, 8])
        for vector, exp_length in zip(net.ws, (24, 12, 36, 72)):
            assert_equal(len(vector), exp_length)
    
    def test_node_weights(self):
        net = NeuralNet(units=[8, 3, 5, 9, 8])
        for i, layer in enumerate(net.ws):
            for j in range(len(layer)):
                net.ws[i][j] = i + j
        for node in net.inputs:
            assert_equal(net.ws[0][node.id], node.id)
        for i, layer in enumerate(net.hidden):
            assert_equal(net.ws[i+1][node.id], i + node.id + 1)

    def test_init_connections(self):
        net = NeuralNet(units=[8, 3, 5, 9, 8])
        first_hidden = net.hidden[0]
        c = count(1)
        for node in net.inputs:
            assert_equal(node.next_layer, first_hidden)
        for layer in net.hidden[:-1]:
            next_layer = net.hidden[next(c)]
            for node in layer:
                assert_equal(node.next_layer, next_layer)
        for node in net.hidden[-1]:
            assert_equal(node.next_layer, net.outputs) 

    def test_feed_forward(self):
        net = NeuralNet(units=[8, 3, 5, 9, 8])
        for i, layer in enumerate(net.ws):
            for j in range(len(layer)):
                net.ws[i][j] = 0.5
        a_node = net.inputs[0]
        a_node.output = 2
        a_node.feed_forward()
        for node in net.hidden[0]:
            assert_equal(node.input, 1)


    def test_backprop_error(self):
        net = NeuralNet(units=[8, 3, 5, 9, 8])
        for i, layer in enumerate(net.ws):
            for j in range(len(layer)):
                net.ws[i][j] = 0.5
        for output in net.outputs:
            output.error = 0.75
        hidden_node = net.hidden[-1][0]
        hidden_node.output = 0.66
        hidden_node.backprop_error()
        assert_almost_equal(hidden_node.error, 0.673, delta=0.01)

    def test_update_weights(self):
        net = NeuralNet(units=[8, 3, 5, 9, 8])
        for i, layer in enumerate(net.ws):
            for j in range(len(layer)):
                net.ws[i][j] = 0.5
        output = net.outputs[0] 
        output.error = 0.25
        output.output = 0.9
        output.update_weights(0.3)
        assert_almost_equal(net.ws[output.layer][output.id], 0.56, delta=0.1)
