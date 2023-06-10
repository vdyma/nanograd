# nanoGrad: Extremely simple autograd engine
## Overview
nanoGrad is a reverse automatic differentiation system. Conceptually, nanoGrad records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the input values and root is the output value. By tracing this graph from root to leaves, you can automatically compute the gradient using the chain rule.

Inspired by Andrej Karpathy's lecture on deep learning, this project has evolved into a comprehensive solution that simplifies training multilayer perceptron (MLP) models while offering a simple visualization of the computation graph.

Key features of nanoGrad:
* Reverse mode autodifferentiation engine for efficient backpropagation.
* Training and optimization of multilayer perceptron (MLP) models.
* Intuitive framework with a simple visualization of the computation graph.
* Implements popular weights initialization functions such as uniform, normal, constant, xavier uniform, xavrier normal and normal truncated.
* Implements popular activation functions such as tanh, exponent, ReLU, sigmoid and logarithm.

TODO:
* Add tests
* Add setup instructions
