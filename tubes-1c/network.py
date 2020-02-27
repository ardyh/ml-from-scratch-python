from random import seed
from random import random
from math import exp
import pandas as pd
import numpy as np

class Network:
  #konstruktor
  def __init__(self, n_inputs, n_hidden, n_outputs=3):
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.weights_ItoH = np.random.uniform(-1, 1, (n_inputs,n_hidden))
    self.weights_HtoO = np.random.uniform(-1, 1, (n_hidden, n_outputs))
    self.pre_activation_H = np.zeros(n_hidden)
    self.post_activation_H = np.zeros(n_hidden)
    self.pre_activation_O = np.zeros(n_outputs)
    self.post_activation_O = np.zeros(n_outputs)

  # Initialize a network - OBSOLETE
  # def initialize_network(self):
  #   seed(1)
  #   self.weight_input_hidden = [[random() for i in range(self.n_inputs+1)] for j in range(self.n_hidden)]
  #   self.weight_hidden_output = [[random() for i in range(self.n_hidden+1)] for j in range(self.n_outputs)]
  
  # Calculate net for an input
  def calculate_net_ItoH(self, sample, node):
    return np.dot(self.data[sample,:], self.weights_ItoH[:, node])
  # Calculate net for a hidden unit
  def calculate_net_HtoO(self, node):
    return np.dot(self.post_activation_H, self.weights_HtoO[:, node])

  # Neuron activation
  def activation(self, x):
  	return 1.0/(1 + np.exp(-x))
  
  # Derivation of activation function
  def activation_deriv(self, x):
    return activation(x) * (1 - activation(x))

  #fit the network to the data
  def fit(self, data, target, epoch_limit=100):
    self.data = data
    self.target = target
    self.epoch_limit = epoch_limit

    len_data = len(data)

    for epoch in range(epoch_limit):
      for instance in range(len_data):
        # From input layer to hidden layer
        
        ## iterate every hidden layer to fill the values
        for hidden_unit in range(self.n_hidden):
          ### calculate the net input
          self.pre_activation_H[hidden_unit] = self.calculate_net_ItoH(instance, hidden_unit)
          ### calculate the activated value
          self.post_activation_H[hidden_unit] = self.activation(self.pre_activation_H[hidden_unit])

        # From hidden layer to output layer
        for output_unit in range(self.n_outputs):
          ### calculate the net input
          self.pre_activation_O[output_unit] = self.calculate_net_HtoO(self.data, self.weights_ItoH, instance, output_unit)
          ### calculate the activated value
          self.post_activation_H[output_unit] = self.activation(self.pre_activation_H[output_unit])

# Testing
net = Network(3,3,3)
net.initialize_network()
# print(net.weight_input_hidden[0])
inputs = [1,1,2,3]  # element [0] is bias
print(net.activate(net.weight_input_hidden[0], inputs))