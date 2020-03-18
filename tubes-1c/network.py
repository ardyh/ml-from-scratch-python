from random import seed
from random import random
from math import exp
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

class Network:
  #konstruktor
  def __init__(self, n_hidden, n_inputs=4, n_outputs=3):
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    
    self.weights_ItoH = np.random.uniform(-1, 1, (n_inputs, n_hidden))
    self.dweights_ItoH = np.zeros((n_inputs, n_hidden))
    
    self.weights_HtoO = np.random.uniform(-1, 1, (n_hidden, n_outputs))
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

    for epoch in range(epoch_limit):
      
      mini_batch_count = 0
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
          for hidden_unit in range(self.n_hidden): #pointer to hidden layer
            for output_unit in range(self.n_outputs): #pointer to output layer
              #### minus sign not merged
              #### formula: -(target_oi - out_oi) * out_oi * (1 - out_oi) * out_hj
              target_o = self.target[instance][output_unit]
              output_o = self.post_activation_O[output_unit]
              output_h = self.post_activation_H[hidden_unit]
              
              self.error_O[output_unit] = -(target_o - output_o) * output_o * (1 - output_o)
              self.weights_HtoO[hidden_unit][output_unit] += self.error_O[output_unit] * output_h
          
          ### update delta-weight from hidden layer
          for input_unit in range(self.n_inputs): #pointer to input layer
            for hidden_unit in range(self.n_hidden): #pointer to hidden layer
              #### formula: sigma_oi(error_oi * w_hj_to_oi) * out_hj * (1 - out_hj) * input_k
              sigma_o = np.dot(error_O, weights_HtoO[ ][])


          #update iterator
          mini_batch_count += 1



# Testing
print('Data Iris')
load, target = load_iris(return_X_y=True)
iris_data = pd.DataFrame(load, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris_data['label'] = pd.Series(target)

net = Network(4, 4)
net.fit(load, 'label', epoch_limit=1)