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

## Usage
You can go throuhg short Jupyter Notebbok `docs/example.ipynb` that covers basic usage.

To run examples, you must install [Graphviz](https://graphviz.org/download/) - open source graph visualization software.

After that install requirements:
```
pip install -r ./requirements.txt
```
Alternatively, you can use `pipenv` for environment management:
```
pipenv install
```

Run Jupyter with and that's it!
```
jupyter-lab
```
Now you're ready to explore nanoGrad!

## Example
This is how you can create and manipulate values. Values accept floats or ints as data and can be assigned a label.
```
a = Value(2.0, label="a")
b = Value(4.0, label="b")
c = a * b
c.label = "c"
d = Value(3.0, label="d")
e = c + d
e.label = "e"
e
```

You can take derivatives of a value with respect to another value.
```
e.backward()
e.grad
```

Visualize the computation graph.
```
e.visualize()
```
![img](/images/example_graph.svg)
