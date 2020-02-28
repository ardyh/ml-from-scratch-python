from random import seed
from random import random
from math import exp
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Network:
  #konstruktor
  def __init__(self, n_inputs, n_hidden, n_outputs=3):
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    
    self.weights_ItoH = np.random.uniform(-1, 1, (n_inputs, n_hidden))
    self.weights_HtoO = np.random.uniform(-1, 1, (n_hidden, n_outputs))
    
    self.dweights_ItoH = np.zeros((n_inputs, n_hidden))
    self.dweights_HtoO = np.zeros((n_hidden, n_outputs))
    
    self.pre_activation_H = np.zeros(n_hidden)
    self.post_activation_H = np.zeros(n_hidden)
    
    self.pre_activation_O = np.zeros(n_outputs)
    self.post_activation_O = np.zeros(n_outputs)
    self.error_O = np.zeros(n_outputs)
  
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

  def one_hot_encode(self, target):
    encoder = OneHotEncoder(sparse=False)
    new_target = target.reshape(len(target), 1)
    target_encode = encoder.fit_transform(new_target)

  #fit the network to the data
  def fit(self, data, target, epoch_limit=100, mini_batch_limit=10):
    self.data = data
    self.target = self.one_hot_encode(target)
    self.epoch_limit = epoch_limit

    len_data = len(data)

    # iterate each epoch
    for epoch in range(epoch_limit):
      
      #iterate each instance
      mini_batch_count = 0
      for instance in range(3):
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
          self.pre_activation_O[output_unit] = self.calculate_net_HtoO(output_unit)
          ### calculate the activated value
          self.post_activation_O[output_unit] = self.activation(self.pre_activation_O[output_unit])
      
        #for debug
        print('INSTANCE:', instance )
        print('WEIGHTS\n', self.weights_ItoH, '\n', self.weights_HtoO)
        print('OUTPUTS\n', self.post_activation_H, '\n', self.post_activation_O)

        # Backpropagation
        ## if already at minibatch limit or at the last instance, update the weight 
        if((mini_batch_count == mini_batch_limit) or (instance == len_data - 1)):
          mini_batch_count = 0
        ## if below minibatch limit, update delta-weight
        else:
          ### update delta-weight from output
          for hidden_unit in range(self.n_hidden):
            for output_unit in range(self.n_outputs):
              #### (Minus sign not merged). Formula: -(target_ok - out_ok) * out_ok * (1 - out_ok) * out_hj
              target_o = self.target[instance][output_unit]
              out_o = self.post_activation_O[output_unit]
              out_h = self.post_activation_H[hidden_]

          ### update delta-weight from hidden layer
          mini_batch_count += 1
        

# Testing
net = Network(3,3,3)
net.initialize_network()
# print(net.weight_input_hidden[0])
inputs = [1,1,2,3]  # element [0] is bias
print(net.activate(net.weight_input_hidden[0], inputs))