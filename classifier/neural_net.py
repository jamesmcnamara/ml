from itertools import count
from math import exp
from operator import ne
from random import random

import numpy as np


class NeuralNet:
    def __init__(self, units=None, eta=1, steps=1e6):
        self.eta = eta
        self.steps = steps
        input_count, *hidden_counts, output_count = units
        self.inputs = [InputNode(eta) for _ in range(input_count)]
        self.hidden_layers = [[Node(eta) for _ in range(nth_layer)]
                              for nth_layer in hidden_counts]
        self.outputs = [OutputNode(eta) for _ in range(output_count)]
        self.layers = [self.inputs] + self.hidden_layers + [self.outputs]
        self.init_connections()
        self.trained = False

    def init_connections(self):
        """
            Connects each node in all non output layers to the i+1th layer
        """
        for layer, next_layer in zip(self.layers, self.layers[1:]):
            for node in layer:
                node.set_next_layer(next_layer)

    def clear_nodes(self):
        """
            Clears out all inputs and outputs from each node in the net
        """
        for layer in self.layers:
            for node in layer:
                node.clear()

    def train(self, data, results):
        """
            Trains this neural network on data and results
        :param data: N*M data vector, where M is the number of input nodes
        :param results: N*K data vector, where K is the number of output nodes
        """
        counter = count()
        while self.has_error(data, results):
            if next(counter) % 10000 == 0:
                print("current is at", next(counter))
            self.stochastic_descent(data, results)
        if self.has_error(data, results):
            print("Iterations maxed out")
        self.trained = True
   
    def has_error(self, data, results):
        """
            Determines if this network produces any errors when predicting the
            results for data
        :param data: N*M data vector, where M is the number of input nodes
        :param results: N*K data vector, where K is the number of output nodes
        :return: True if any elements in data are misclassified
        """
        for row, result in zip(data, results):
            if any(map(ne, row, self.predict(row))):
                return True
        return False

    def stochastic_descent(self, data, results):
        """
            Runs one cycle of the backprop error algorithm on data
            and results
        :param data: N*M data vector, where M is the number of input nodes
        :param results: N*K data vector, where K is the number of output nodes
        """
        for row, result in zip(data, results):
            #print(row)
            self.predict(row)
            self.backprop(result)

    def predict(self, row):
        """
            Runs the feed forward algorithm on the given row
        :param row: 1*M data vector, where M is the number of input nodes
        :return: predicted output for row
        """
        self.clear_nodes()
        for element, input_node in zip(row, self.inputs):
            input_node.set_input(element)
        for layer in self.layers:
            for node in layer:
                node.feed_forward(print_hidden=self.trained)
        return [0 if out.output < 0.8 else 1 for out in self.outputs] 

    def backprop(self, result):
        """
            Backpropogates the error of this net given that the result should
             be result
        :param result: 1*K data vector, where K is the number of output nodes
        """
        for resultum, output in zip(result, self.outputs):
            #print("output errors")
            output.calc_error(resultum)
        for layer in (self.hidden_layers[::-1] + [self.inputs]):
            #print("new layer errors and weights")
            for node in layer:
                node.calc_error()
                node.update_weights()


class Node:
    def __init__(self, eta):
        self.eta = eta
        self.output = self.input = self.error = 0
        self.ws = self.next_layer = None
        self.bias = random()

    def set_next_layer(self, layer):
        """
            Initializes the reference to the next layer, and creates a random
            weight vector
        :param layer: list of nodes that belong to the next layer
        """
        self.next_layer = layer
        self.ws = np.random.random(len(layer))
        #self.ws = np.zeros(len(layer))
        #self.ws.fill(.1)

    def set_input(self, input):
        """
            Set the input to this node
        :param input: float value
        """
        raise AttributeError("Hidden node's input must be set by other nodes")
        
    def clear(self):
        """
            clear the input, output and error of this node
        """
        self.input = self.output = self.error = 0

    def calc_output(self):
        """
            Set the output value of this node using it's input field
        """
        #print("input was", self.input, "bias", self.bias)
        self.output = 1 / (1 + exp(-(self.input + self.bias)))
        #print("output is", self.output)

    def feed_forward(self, print_hidden=False):
        """
            Adds to the inputs of each node in the next layer the output
            of this node times the weight from this to that node
        :param #print_hidden: if true, #print to the console the output value
            of this hidden node
        """
        self.calc_output()
        if print_hidden and self.output != self.input:
            print(self.output)
        for wij, node in zip(self.ws, self.next_layer):
            node.input += (self.output * wij)
        #print("weights are", self.ws)
    
    def calc_error(self, *args):
        """
            Calculates the error of this node based on the error of nodes in the
            next layer
        :param args: unused here
        """
        self.error = (self.output * (1 - self.output) *
                      sum(node.error * wij
                          for wij, node in zip(self.ws, self.next_layer)))
        #print("error is", self.error)

    def update_weights(self):
        """
            Based on the error of each node in the next layer and this
            node's input, updates the weight from this node to that node
        """
        for j, node in enumerate(self.next_layer):
            self.ws[j] += self.eta * (node.error * self.output)
        #print("bias was", self.bias, "eta is", self.eta)
            self.bias += (self.error * self.eta)
        #print("bias became", self.bias)
        #print("weights are", self.ws)


class InputNode(Node):
    def set_input(self, input):
        """
            OVERRIDE: Input nodes can set their own input
        :param input: a float value to to set as input
        """
        self.input = input 
    
    def calc_output(self):
        """
            OVERRIDE: Input node's outputs are their inputs
        """
        self.output = self.input

    def calc_error(self):
        """
            OVERRIDE: input nodes have no error
        """
        self.error = 0


class OutputNode(Node):
    def set_next_layer(self, layer):
        """
            OVERRIDE: output nodes have no next layer
        :param layer: next layer
        """
        raise AttributeError("Output nodes do not have a next layer")

    def calc_error(self, result, *args):
        """
            OVERRIDE: The error of an output node is directly dependent on
            the output result
        :param result: bit (0 or 1)
        :param args: unused
        """
        self.error = self.output * (1 - self.output) * (result - self.output)
        #print("output error is", self.error)

    def feed_forward(self, print_hidden=False):
        """
            OVERRIDE: Output nodes only need to calculate their input
        :param #print_hidden: unused
        """
        self.calc_output()

