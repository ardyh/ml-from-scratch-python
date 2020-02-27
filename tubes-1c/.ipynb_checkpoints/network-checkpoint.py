from random import seed
from random import random
from math import exp

class Network:
  #konstruktor
  def __init__(self, n_inputs, n_hidden, n_outputs):
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.input_weight_hidden = list()
    self.output_weight_hidden = list()

  # Initialize a network
  def initialize_network(self):
    seed(1)
    self.input_weight_hidden = [[random() for i in range(self.n_inputs+1)] for j in range(self.n_hidden)]
    self.output_weight_hidden = [[random() for i in range(self.n_hidden+1)] for j in range(self.n_outputs)]
  
  # Calculate net for an input
  def calculate_net(self, weights, inputs):
    net = 0
    for i in range(len(weights)):
      net += weights[i] * inputs[i]
    return net

  # Neuron activation
  def activate(self, weights, inputs):
  	return 1.0 / (1.0 + exp(-1*self.calculate_net(weights, inputs)))

# Testing
net = Network(3,3,3)
net.initialize_network()
# print(net.input_weight_hidden[0])
inputs = [1,1,2,3]  # element [0] is bias
print(net.activate(net.input_weight_hidden[0], inputs))